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

COLS_TO_REMOVE = [
    "CANCER_TYPE_ACRONYM",
    "OTHER_PATIENT_ID",
    "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
    "HISTORY_NEOADJUVANT_TRTYN",
    "PATH_M_STAGE",
    "ICD_O_3_SITE", # removed because is the same as ICD_10
    "ICD_O_3_",
    "DAYS_LAST_FOLLOWUP",              # follow-up time after diagnosis (future info)
    "FORM_COMPLETION_DATE",            # administrative metadata, not predictive
    "INFORMED_CONSENT_VERIFIED",       # administrative, no biological meaning
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",  # recurrence event → direct leakage
    "PERSON_NEOPLASM_CANCER_STATUS",   # disease status at follow-up → leakage
    "IN_PANCANPATHWAYS_FREEZE",        # technical/analysis flag, not biological
    "OS_STATUS",                       # overall survival outcome → leakage
    "OS_MONTHS",                       # overall survival time → leakage
    "DSS_STATUS",                      # disease-specific survival outcome → leakage
    "DSS_MONTHS",                      # disease-specific survival time → leakage
    "DFS_STATUS",                      # disease-free survival outcome → leakage
    "DFS_MONTHS",                      # disease-free survival time → leakage
    "PFS_STATUS",                      # progression-free survival outcome → leakage
    "PFS_MONTHS"                       # progression-free survival time → leakage
]


# this shouldn't exist, I should change clinical preprocessor to 
# find these on its own because I don't have any ordinal categories
#  as of right now
CATEGORICS_COLS = ['SUBTYPE',
                    'ETHNICITY', 
                    "ICD_10", 
                    "ICD_O_3_HISTOLOGY", 
                    "PRIOR_DX", 
                    "RACE", 
                    "RADIATION_THERAPY", 
                    "GENETIC_ANCESTRY_LABEL"
]

# Stability selection parameters
N_BOOTSTRAPS = 100
FPR_ALPHA = 0.05
STABILITY_THRESHOLD = 0.95
