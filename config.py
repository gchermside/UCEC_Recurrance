# config.py

# Directories
DATA_DIR = 'data/'
MODEL_DIR = 'models/'

# Data file paths
X_TRAIN_PATH = DATA_DIR + 'X_train.pkl'
Y_TRAIN_PATH = DATA_DIR + 'y_train.pkl'
X_TEST_PATH = DATA_DIR + 'X_test.pkl'
Y_TEST_PATH = DATA_DIR + 'y_test.pkl'
FEATURE_NAMES = DATA_DIR + "feature_names.pkl"

# Model paths
SVC_MODEL_PATH = MODEL_DIR + 'SVC_no_LASSO.pkl'
RF_MODEL_PATH = MODEL_DIR + 'random_forest_model.pkl'
LASSO_MODEL_PATH = MODEL_DIR + 'lasso_model.pkl'

# Experiment metadata
SEED = 100

# columns with more than this percent nulls are removed
MAX_NULL_VALS = 0.3 
