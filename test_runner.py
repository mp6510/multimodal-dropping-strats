import mask_freezing_1DCNN
import json, os

'''
	ATTENTION! ATTENTION! This test configuration makes use of PyTorch and 
	the Metal API framework, requiring the use of an Apple M1 (or higher) chip.
	If run on hardware that does not have this, it will default to running
	operations on the CPU. Please consider lowering the batch_size in this case.
'''

# The number of iterations will be |t_w|*|t_s|*|d_s|*|k|*|seed_values|*|num_epochs|
# It will periodically output results as a tsv file and update it throughout the run

# TEST CONFIGURATION
mask_freezing_1DCNN.t_w_list = [1500,2000,2500,3000,3500,4000,4500,5000] # Time window values (milliseconds) {1500,2000,2500,3000,3500,4000,4500,5000}
mask_freezing_1DCNN.t_s_list = [20,30,40,50] # Time step values to test {20,30,40,50}
mask_freezing_1DCNN.d_s_list = [1,2,3,4] # Dropping strategies to test {'A': 1,'B': 2,'C': 3,'D': 4}
mask_freezing_1DCNN.k = 10 # For k-Fold cross validation
mask_freezing_1DCNN.seed_values = [7,777,144000] # Seed values used to create model in each k-fold, can be any set of integers
mask_freezing_1DCNN.verbose = False # Output validation statistics at each epoch step

# DATA CONFIGURATION 
mask_freezing_1DCNN.data_dir = "confusion_data" # Change data directory
mask_freezing_1DCNN.embedding_dim = 155 # Set size of ChatGPT ada embeddings, via determined through PCA elbow test
mask_freezing_1DCNN.gendered = "Neutral" # {Male, Female, Neutral}
mask_freezing_1DCNN.binary_type = "Balance" # {Balance, More, Less}
mask_freezing_1DCNN.binary_task = True # If set to False, will use imbalanced four class distribution

# MODEL CONFIGURATION
mask_freezing_1DCNN.num_epochs = 20
mask_freezing_1DCNN.batch_size = 300 # Lower this number if running on CPU
mask_freezing_1DCNN.learning_rate = 0.01 # A scheduler also automatically adjust this
mask_freezing_1DCNN.use_lr_scheduler = True
mask_freezing_1DCNN.use_layer_freezing = True # See if there is a difference in freezing weights
mask_freezing_1DCNN.save_best_model = False # Will output best model to corresponding directory

# SAVE TEST CONFIG
config_dict = {
	't_w_list': mask_freezing_1DCNN.t_w_list,
	't_s_list': mask_freezing_1DCNN.t_s_list,
	'd_s_list': mask_freezing_1DCNN.d_s_list,
	'k': mask_freezing_1DCNN.k,
	'seed_values': mask_freezing_1DCNN.seed_values,
	'verbose': mask_freezing_1DCNN.verbose,
	'data_dir': mask_freezing_1DCNN.data_dir,
	'embedding_dim': mask_freezing_1DCNN.embedding_dim,
	'gendered': mask_freezing_1DCNN.gendered,
	'binary_type': mask_freezing_1DCNN.binary_type,
	'binary_task': mask_freezing_1DCNN.binary_task,
	'num_epochs': mask_freezing_1DCNN.num_epochs,
	'batch_size': mask_freezing_1DCNN.batch_size,
	'learning_rate': mask_freezing_1DCNN.learning_rate,
	'use_lr_scheduler': mask_freezing_1DCNN.use_lr_scheduler,
	'use_layer_freezing': mask_freezing_1DCNN.use_layer_freezing,
	'save_best_model': mask_freezing_1DCNN.save_best_model,
}

# OUTPUT TEST CONFIG
with open(os.path.join(mask_freezing_1DCNN.data_dir, f"test_config_{mask_freezing_1DCNN.start_time_file_readable}.json"), 'w') as file:
	json.dump(config_dict, file)

# Run the test loop
mask_freezing_1DCNN.run_test()