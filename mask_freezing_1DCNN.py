import os
import ast
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import time

# Get the current time when the program starts
start_time = time.time()
start_time_human_readable = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
start_time_file_readable = time.strftime('%m-%d_%H-%M-%S', time.localtime(start_time))
print(f'Program started at: {start_time_human_readable}')

# Can be overwritten by the test runner; these are default values
t_w_list = [1500,2000,2500,3000,3500,4000,4500,5000] # Time windows (milliseconds)
t_s_list = [5,10,15,20] # Time steps
d_s_list = [1,2,3,4] # Modality dropping strategies
seed_values = [7,144000] # Used to randomize model instantiation weights
k = 10 # For k-Fold cross validation
embedding_dim = 155 # The dimensions to reduce ChatGPT ada embeddings, determined through PCA elbow test
gendered = 'Neutral' # {Male, Female, Neutral}
binary_task = True
binary_type = 'Balance' # {More, Less}
num_epochs = 20
batch_size = 300
learning_rate = 0.01
use_lr_scheduler = True
use_layer_freezing = True
verbose = True
save_best_model = False
data_dir = "confusion_data"
alpha = 0.05 # For rejecting the null hypothesis

def run_test():
    # Test configuration
    default_m_dim = 12 # The original number of modalities from paper
    freeze_val = -10 # Make sure this matches the value in process_data.py
    seed = 777 # For reproducibility
    ablation_study = (False, [f"m{i}" for i in range(1,default_m_dim+1)])
    # ablation_study = (True, ["m7","m9"])
    shuffle = True

    # Program parameters
    new_msg_stats = True
    run_test_set = False

    # All the configurations used for INTERSPEECH 2025 paper
    d_s_list.insert(0, 0) # Always run the clean data

    # Directory and output file information
    results_dir = "results"
    model_dir = "models"
    results_file = f"results_1DCNN_FREEZING_{use_layer_freezing}_{gendered.upper()}{'_BINARY' if binary_task else '_MULTI'}{f'_{ablation_study[1]}' if ablation_study[0] else ''}_{start_time_file_readable}.tsv"
    epochs_file = f"epochs_1DCNN_FREEZING_{use_layer_freezing}_{gendered.upper()}{'_BINARY' if binary_task else '_MULTI'}{f'_{ablation_study[1]}' if ablation_study[0] else ''}_{start_time_file_readable}.tsv"
    parent_dir = os.getcwd()
    participants_df = pd.read_csv(os.path.join(parent_dir,data_dir,"participant_info.tsv"), sep="\t")

    # 1D CNN classifier hyperparameters
    input_size = len(ablation_study[1])
    num_classes = 2 if binary_task else 4 

    # Define test groups to remove from the development data
    test_groups = ["17C", "22N", "25C"]

    # Convert labels to a binary problem
    def convert_labels_to_binary(Y_df, b_type):
        if b_type == "More":
            # Merge Slightly class with Not At All class
            Y_df.loc[Y_df["Rating"] == 1, "Rating"] = 0
            Y_df.loc[Y_df["Rating"] > 1, "Rating"] = 1

        elif b_type == "Less":
            # Remove Slightly class
            Y_df = Y_df[Y_df["Rating"] != 1]
            Y_df.loc[Y_df["Rating"] > 1, "Rating"] = 1
        
        elif b_type == "Balance":
            # Merge Slightly class with Very and Extremely class
            # Then balance the dataframe by removing Slightly instances
            Y_df['Slightly'] = (Y_df['Rating']==1) # Tag Slightly class
            Y_df.loc[Y_df["Rating"] > 1, "Rating"] = 1 # Merge with Very and Extremely
            groups = Y_df['Group'].unique().tolist() # Get development groups
            dev_groups = [g for g in groups if g not in test_groups]
            for i in range(1000):
                curr_group = dev_groups[i % len(dev_groups)] # Cycle through dev groups
                slightly_indices = Y_df[(Y_df['Slightly'])&(Y_df['Group']==curr_group)].index
                if not slightly_indices.empty:
                    Y_df = Y_df.drop(slightly_indices[0]) # Drop the Slightly instance
                if len(Y_df[Y_df['Rating']==0]) == len(Y_df[Y_df['Rating']==1]):
                    # We have balanced the dataset
                    break
        return Y_df

    # Add gender information to the labels
    def add_gender_to_labels(Y_df):
        genders = [participants_df[(participants_df["Group"] == group) & (participants_df["Role"] == "Instructor")]["Sex"].iloc[0] for group in Y_df["Group"].tolist()]
        Y_df["Gender"] = genders
        return Y_df

    # Set the seed for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def remove_hooks(model):
        for module in model.modules():
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()

    # Define the novel model architecture
    class Masked1DCNN(nn.Module):
        def __init__(self, num_channels, num_timesteps, num_classes, embedding_dim):
            super(Masked1DCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            
            self._to_linear = None
            self._initialize_to_linear(num_channels, num_timesteps)
            
            self.fc1 = nn.Linear(self._to_linear + embedding_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
        
        def _initialize_to_linear(self, num_channels, num_timesteps):
            # Forward pass through convolutional layers to determine the size
            x = torch.rand(1, num_channels, num_timesteps)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self._to_linear = x.numel()

        def forward(self, x, mask, embeddings):
            x = x * mask  # Apply mask to the input
            x = x.permute(0, 2, 1) # Conv1d (batch_size, features, sequence_length)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten

            # COMBINE WITH ADA EMBEDDINGS
            combined_features = torch.cat((x, embeddings), dim=1)
            x = F.relu(self.fc1(combined_features))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def register_hooks(self):
            def hook_fn_conv1(module, grad_input, grad_output):
                grad = grad_input[0]
                if grad is not None:
                    # Adjust mask dimensions to match grad dimensions
                    mask_resized = F.interpolate(self.mask, size=(grad.size(2),), mode='nearest')
                    # Create an empty tensor with the same shape as grad
                    mask_expanded = torch.zeros_like(grad)
                    # Manually assign the resized mask to the expanded mask tensor
                    for i in range(grad.size(1)):
                        mask_expanded[:, i, :] = mask_resized[:, i % self.mask.size(1), :]
                    grad = grad * mask_expanded
                return (grad,)

            def hook_fn_conv2(module, grad_input, grad_output):
                grad = grad_input[0]
                if grad is not None:
                    # Adjust mask dimensions to match grad dimensions
                    mask_resized = F.interpolate(self.mask, size=(grad.size(2),), mode='nearest')
                    # Create an empty tensor with the same shape as grad
                    mask_expanded = torch.zeros_like(grad)
                    # Manually assign the resized mask to the expanded mask tensor
                    for i in range(grad.size(1)):
                        mask_expanded[:, i, :] = mask_resized[:, i % self.mask.size(1), :]
                    grad = grad * mask_expanded
                return (grad,)

            self.conv1.register_full_backward_hook(hook_fn_conv1)
            self.conv2.register_full_backward_hook(hook_fn_conv2)

        def set_mask(self, mask):
            self.mask = mask

    # Set device settings to use Metal for Apple M1 MAX devices
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 1 John 3:9, to God be the glory, forever and ever
    set_seed(seed)
    best_overall_score = 0.4
    best_overall_scores = []

    print(f"Beginning loop for file {results_file}...")
    epoch_values = []
    results = []
    # Begin test configuration loop, reading in different time windows
    for t_w in t_w_list:
        # READ IN ADA EMBEDDINGS
        ada_Y_df = pd.read_csv(os.path.join(parent_dir,data_dir,f"Y_{t_w}milli_embedding.tsv"), sep='\t')
        for t_dim in t_s_list:
            baseline_res = None # For computing p-values to determine statistically significant values this does not seem to 
            for d_s in d_s_list:
                # Modality dropping strategies, including baseline
                X_df = pd.read_csv(os.path.join(parent_dir,data_dir,f"X_{t_w}milli_{t_dim}dim_{d_s}strat_confusion.tsv"), sep='\t')
                Y_df = pd.read_csv(os.path.join(parent_dir,data_dir,f"Y_{t_w}milli_{t_dim}dim_{d_s}strat_confusion.tsv"), sep='\t')
                XY_df = pd.concat([X_df,Y_df], axis=1)
                XY_df = pd.concat([XY_df,ada_Y_df["Embedding"]], axis=1)
                
                if run_test_set:
                    # TEST DATA
                    XY_df = XY_df[XY_df["Group"].isin(test_groups)]
                else:    
                    # DEVELOPMENT DATA
                    XY_df = XY_df[~XY_df["Group"].isin(test_groups)]
                
                # Data configurations
                if gendered != "Neutral":
                    XY_df = add_gender_to_labels(XY_df)
                    XY_df = XY_df[XY_df["Gender"] == gendered]
                
                if binary_task:
                    XY_df = convert_labels_to_binary(XY_df, binary_type)
                
                # Acquire ChatGPT ada text embeddings
                embeddings = np.array([ast.literal_eval(e) for e in XY_df["Embedding"]])
                # Applying PCA to reduce the dimensionality
                pca = PCA(n_components=embedding_dim)
                reduced_embeddings = pca.fit_transform(embeddings)
                reduced_embeddings_tensor = torch.tensor(reduced_embeddings, dtype=torch.float32)

                # Convert to numpy, selecting the proper columns
                select_cols = []
                for i in range(1,t_dim+1):
                    for m in ablation_study[1]:
                        select_cols.append(f"{m}_t{i}")
                train_X = XY_df[select_cols].to_numpy()
                train_y = XY_df["Rating"].to_numpy()

                if new_msg_stats:
                    print(f"FINAL DEV SET STATS: {{len(train_X): {len(train_X)}, len(train_y): {len(train_y)}, reduced_embeddings_tensor.size(0): {reduced_embeddings_tensor.size(0)}, value_counts: \n{XY_df['Rating'].value_counts()}}}")
                    new_msg_stats = False

                # Convert to PyTorch tensors
                # reshape input for (instances, timesteps, modalities)
                inputs_tensor = torch.tensor(train_X.reshape(train_X.shape[0],t_dim,input_size), dtype=torch.float32)
                targets_tensor = torch.tensor(train_y, dtype=torch.long)

                # Assuming targets_tensor contains the labels (0s and 1s)
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets_tensor), y=targets_tensor.numpy())
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

                print(f"Evaluating t_w = {t_w}, t_s = {t_dim}, d_s = {d_s}...")
                kf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
                fold_res = [] # Used for baseline comparison
                for fold, (train_index, val_index) in enumerate(kf.split(inputs_tensor, targets_tensor)):
                    if verbose:
                        print(f"Beginning of k={fold+1} fold:")                
                    
                    # Split data
                    train_inputs, val_inputs = inputs_tensor[train_index], inputs_tensor[val_index]
                    train_embeddings, val_embeddings = reduced_embeddings_tensor[train_index], reduced_embeddings_tensor[val_index]
                    train_targets, val_targets = targets_tensor[train_index], targets_tensor[val_index]

                    # Create DataLoader for train and validation sets
                    train_dataset = TensorDataset(train_inputs, train_embeddings, train_targets)
                    val_dataset = TensorDataset(val_inputs, val_embeddings, val_targets)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    for j, seed_j in enumerate(seed_values):
                        if verbose:
                            print(f"\tBeginning of SEED [{j+1}/2] for {{t_w: {t_w}, t_s: {t_dim}, d_s: {d_s}}}")
                        set_seed(seed_j)
                        model = Masked1DCNN(input_size, t_dim, num_classes, embedding_dim)
                        model.to(device)  # Ensure model is on the right device
                        criterion = nn.CrossEntropyLoss(weight=class_weights)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=verbose)

                        # Register hooks only if using layer freezing
                        if use_layer_freezing:
                            model.register_hooks()

                        # Training loop
                        for epoch in range(num_epochs):
                            # Used to discover the best ov with curr config
                            model.train() # Sets model in a mode to have its weights adjusted
                            for inputs, batch_embeddings, targets in train_loader:
                                optimizer.zero_grad()

                                # Create instance-specific mask freezing for training data
                                train_masks = torch.ones_like(inputs)
                                for i in range(inputs.size(0)):
                                    mask = torch.ones(t_dim, input_size)
                                    # Get the coordinates of the freeze value
                                    coordinates = torch.nonzero(inputs[i] == freeze_val, as_tuple=False)
                                    coordinates_list = [tuple(coord) for coord in coordinates]
                                    for mask_coord in coordinates_list:
                                        mask[mask_coord] = 0 # FREEEZE !!
                                    train_masks[i] = mask
                                
                                # Move batch over to the GPU
                                inputs, batch_embeddings, targets, train_masks = inputs.to(device), batch_embeddings.to(device), targets.to(device), train_masks.to(device)

                                # Set the masks for the model
                                model.set_mask(train_masks.permute(0, 2, 1)) # permute to match (instances, features, timesteps)

                                # Forward pass, with the freeze masks
                                outputs = model(inputs, train_masks, batch_embeddings)
                                loss = criterion(outputs, targets)
                                t_loss = loss.item()
                                loss.backward()

                                # Only perform an optimization step after the gradients
                                # have been properly FROZEN !!
                                optimizer.step()

                            # Validation
                            model.eval()
                            all_targets = []
                            all_predictions = []
                            v_loss = 0
                            with torch.no_grad():
                                for inputs, batch_embeddings, targets in val_loader:
                                    # Create instance-specific mask freezing for validation data
                                    val_masks = torch.ones_like(inputs)
                                    for i in range(inputs.size(0)):
                                        mask = torch.ones(t_dim, input_size)
                                        coordinates = torch.nonzero(inputs[i] == freeze_val, as_tuple=False)
                                        coordinates_list = [tuple(coord) for coord in coordinates]
                                        for mask_coord in coordinates_list:
                                            mask[mask_coord] = 0 # FREEEZE !!
                                        val_masks[i] = mask

                                    # Move batch over to the GPU
                                    inputs, batch_embeddings, targets, val_masks = inputs.to(device), batch_embeddings.to(device), targets.to(device), val_masks.to(device)

                                    outputs = model(inputs, val_masks, batch_embeddings)
                                    _, predicted = torch.max(outputs.data, 1)
                                    all_targets.extend(targets.cpu().numpy())
                                    all_predictions.extend(predicted.cpu().numpy())
                                    loss = criterion(outputs, targets)
                                    v_loss += loss.item() # Aggregate v_loss in epoch
                            if use_lr_scheduler:
                                scheduler.step(v_loss) # Used for the learning rate scheduler
                            
                            # Collect validation metrics
                            val_accuracy = accuracy_score(all_targets, all_predictions)
                            if binary_task:
                                val_precision = precision_score(all_targets, all_predictions, zero_division=0)
                                val_recall = recall_score(all_targets, all_predictions, zero_division=0)
                                val_f1 = f1_score(all_targets, all_predictions, zero_division=0)
                            else:
                                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)

                            # Custom performance metric, operating value, also known as "OV"
                            curr_score = (val_accuracy*0.1)+(val_precision*0.5)+(val_recall*0.2)+(val_f1*0.2)
                            epoch_values.append({
                                't_w': t_w,
                                't_s': t_dim,
                                'd_s': d_s,
                                'k': fold,
                                'seed_j': seed_j,
                                'val_acc': val_accuracy,
                                'val_prec': val_precision,
                                'val_recall': val_recall,
                                'val_f1': val_f1,
                                'val_ov': curr_score,
                                'epoch': epoch,
                                'train_loss': t_loss,
                                'val_loss': v_loss,
                            })
                            if verbose:
                                print(f'\t\tEpoch [{epoch+1}/{num_epochs}], Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}, Val Acc: {val_accuracy*100:.4f}%')

                            if curr_score > best_overall_score:
                                best_overall_score = curr_score
                                best_overall_scores.append({
                                    't_w': t_w,
                                    't_s': t_dim,
                                    'd_s': d_s,
                                    'k': fold,
                                    'seed_j': seed_j,
                                    'epoch': epoch,
                                    'val_acc': val_accuracy,
                                    'val_prec': val_precision,
                                    'val_recall': val_recall,
                                    'val_f1': val_f1,
                                    'train_loss': t_loss,
                                    'val_loss': v_loss,
                                    'val_ov': curr_score,
                                })
                                if save_best_model:
                                    # Save the model
                                    torch.save(model.state_dict(), os.path.join(parent_dir,model_dir,f'best_model_{t_w}tw_{t_dim}ts_{d_s}ds_{fold}k_{seed_j}seed_{start_time_file_readable}.pth'))
                    
                    # Capture all the fold results for use in calculating the p-value
                    epochs_df = pd.DataFrame(epoch_values)
                    k_fold_cond = (epochs_df['t_w'] == t_w) & (epochs_df['t_s'] == t_dim) & (epochs_df['d_s'] == d_s) & (epochs_df['k'] == fold)
                    fold_res += epochs_df.loc[k_fold_cond].to_dict(orient='records')
                
                res_df = pd.DataFrame(fold_res) # Best results from each fold
                if d_s == 0:
                    baseline_res = res_df.copy() # Best baseline per k
                else:
                    baseline_res.sort_values(by=['k', 'seed_j', 'epoch'], ascending=True, inplace=True)
                    res_df.sort_values(by=['k', 'seed_j', 'epoch'], ascending=True, inplace=True)
                    delta_arr = (res_df['val_acc'] - baseline_res['val_acc']).to_numpy()
                    ov_delta_arr = (res_df['val_ov'] - baseline_res['val_ov']).to_numpy()
                    _, p_val = wilcoxon(delta_arr[delta_arr != 0], alternative='greater', method='exact')
                    _, ov_p_val = wilcoxon(ov_delta_arr[ov_delta_arr != 0], alternative='greater', method='exact')
                res_df.sort_values(by='val_ov', ascending=False, inplace=True)
                baseline_res.sort_values(by='val_ov', ascending=False, inplace=True)
                delta = res_df['val_acc'].iloc[0] - baseline_res['val_acc'].iloc[0]
                ov_delta = res_df['val_ov'].iloc[0] - baseline_res['val_ov'].iloc[0]
                # Save the best configuration for the three values
                results.append({
                    't_w': t_w,
                    't_dim': t_dim,
                    'd_s': d_s,
                    'k': res_df['k'].iloc[0],
                    'seed_j': res_df['seed_j'].iloc[0],
                    'epoch': res_df['epoch'].iloc[0],
                    'acc': f"{res_df['val_acc'].iloc[0]:.4f}",
                    'delta': 0 if d_s == 0 else f"{delta:.4f}",
                    'prec': f"{res_df['val_prec'].iloc[0]:.4f}",
                    'rec': f"{res_df['val_recall'].iloc[0]:.4f}",
                    'f1': f"{res_df['val_f1'].iloc[0]:.4f}",
                    'ov': f"{res_df['val_ov'].iloc[0]:.4f}",
                    'ov_delta': 0 if d_s == 0 else f"{ov_delta:.4f}",
                    'p_val': 0 if d_s == 0 else p_val,
                    'p_ov_val': 0 if d_s == 0 else ov_p_val,
                })
                print(f'—— Final results of t_w: {t_w}, t_s: {t_dim}, d_s: {d_s} ——')
                print(f'{{acc: {results[-1]["acc"]}, delta: {results[-1]["delta"]}, ov: {results[-1]["ov"]}, ov_delta: {results[-1]["ov_delta"]}, prec: {results[-1]["prec"]}, p_val: {results[-1]["p_val"]}, p_ov_val: {results[-1]["p_ov_val"]}}}')

                # Save losses and performance as a checkpoint for the current configuration
                epochs_df = pd.DataFrame(epoch_values)
                epochs_df.to_csv(os.path.join(parent_dir,results_dir,epochs_file), sep='\t', index=False)
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(parent_dir,results_dir,results_file), sep='\t', index=False)
                best_df = pd.DataFrame(best_overall_scores)
                best_df.to_csv(os.path.join(parent_dir,results_dir,f'best_overall_{results_file}'), sep='\t', index=False)

                # Calculate the elapsed time
                ds_end_time = time.time()
                elapsed_time = ds_end_time - start_time
                print(f"d_s = {d_s} finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ds_end_time))}")
                print(f"Elapsed time: {((elapsed_time / 60)%60):.2f} minutes")

            # Calculate the elapsed time
            ts_end_time = time.time()
            elapsed_time = ts_end_time - start_time
            print(f"t_s = {t_dim}, d_s = {d_s} finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_end_time))}")
            print(f"Elapsed time: {((elapsed_time / 60)%60):.2f} minutes")

        epochs_df = pd.DataFrame(epoch_values)
        epochs_df.to_csv(os.path.join(parent_dir,results_dir,epochs_file), sep='\t', index=False)

        # Calculate the elapsed time
        tw_end_time = time.time()
        elapsed_time = tw_end_time - start_time
        print(f"t_w = {t_w}, t_s = {t_dim}, d_s = {d_s} finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tw_end_time))}")
        print(f"Elapsed time: {((elapsed_time / 60)%60):.2f} minutes")

    # Save all results to TSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(parent_dir,results_dir,results_file), sep='\t', index=False)
    epochs_df = pd.DataFrame(epoch_values)
    epochs_df.to_csv(os.path.join(parent_dir,results_dir,epochs_file), sep='\t', index=False)
    
    # Output best overall configurations
    best_df = pd.DataFrame(best_overall_scores)
    best_df.to_csv(os.path.join(parent_dir,results_dir,f'best_overall_{results_file}'), sep='\t', index=False)

    # Get the current time when the program ends
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Program finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed time: {((elapsed_time / 60)%60):.2f} minutes")

if __name__ == "__main__":
    run_test()