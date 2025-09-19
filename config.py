# config.py

# Directories
DATA_DIR = 'data/'
MODEL_DIR = 'models/'

# Data file paths
X_TRAIN_PATH = DATA_DIR + 'X_train.pkl'
Y_TRAIN_PATH = DATA_DIR + 'y_train.pkl'
X_TEST_PATH = DATA_DIR + 'X_test.pkl'
Y_TEST_PATH = DATA_DIR + 'y_test.pkl'
FEATURE_NAMES = DATA_DIR + 'feature_names.pkl'

# Model paths
SVC_NO_LASSO_MODEL_PATH = MODEL_DIR + 'SVC_no_LASSO.pkl'
SVC_WITH_LASSO_MODEL_PATH = MODEL_DIR + 'SVC_with_LASSO.pkl'
RF_MODEL_PATH = MODEL_DIR + 'random_forest_model.pkl'
LASSO_MODEL_PATH = MODEL_DIR + 'lasso_model.pkl'
LR_MODEL_PATH = MODEL_DIR + 'logistic_regression.pkl'
XGB_MODEL_PATH = MODEL_DIR + 'xgboost_model_with_LASSO.pkl'

# Experiment metadata
SEED = 100

# columns with more than this percent nulls are removed
MAX_NULL_VALS = 0.25

# columns with more than this percent of the same value are removed
UNIFORM_THRESHOLD = 0.99

# Threshold for feature selection based on correlation
CORRELATION_THRESHOLD = 0.9

# Features with variance below this value are removed
VARIANCE_THRESHOLD = 1e-5

# Genes from https://pmc.ncbi.nlm.nih.gov/articles/PMC7565375/ 
# and https://pmc.ncbi.nlm.nih.gov/articles/PMC9929804/ FIXME: look more into this later
LITERATURE_GENES = set([
    "MLH1", "MSH2", "MSH6", "PMS2", "PTEN", "POLD1", "POLE", "NTHL1", "MUTYH", "BRCA1", "GINS4", "ESR1"
])
