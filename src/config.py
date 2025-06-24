import pathlib

# Paths
BASE_DIR = pathlib.Path(__file__).parents[1]
SRC_DIR = BASE_DIR / pathlib.Path('src')
MODELS_DIR = BASE_DIR / pathlib.Path('model_data')
JSON_LOG_DIR = BASE_DIR / pathlib.Path('json_logs')
# Reset me
MIMIC_IV_DATA_DIR = pathlib.Path('/path/to/unzipped/mimic-iv-2.2')
BASE_DATA_DIR = BASE_DIR / pathlib.Path('data')
RAW_DATA_DIR = BASE_DATA_DIR / pathlib.Path('raw_extracted_data')
LABEL_PATH = BASE_DATA_DIR / pathlib.Path('patient_cohort.csv')
UNNORMALIZED_DATASET_PATH = BASE_DATA_DIR / pathlib.Path('hypotension_unnormalized.csv')
NORMALIZED_DATASET_PATH = BASE_DATA_DIR / pathlib.Path('hypotension_normalized.csv')
NORMALIZATION_PARAMS_PATH = BASE_DATA_DIR / pathlib.Path('normalization_params.csv')
ANALYSIS_PATH = BASE_DIR / pathlib.Path('analysis')


# Postgres
# Reset me
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'resetpassword'
POSTGRES_HOST = '127.0.0.1'
POSTGRES_PORT = '5432'

# Features
DISCRETE_FEATURES = ['MAP','DBP', 'SBP', 'urine', 'ALT', 'AST', 'pO2', 'lactate', 'serum_creatinine']
CLASS_FEATURES = {
    'GCS' : {'nof_classes' : 13, 'median' : 11, 'average distance' : 4.065, 'mean': 8.918},
    'FiO2' : {'nof_classes' : 10, 'median' : 4, 'average distance' : 2.048, 'mean': 4.303},
    'bolus' : {'nof_classes' : 3, 'median' : 1, 'average distance' : 0.744, 'mean': 1.269},
    'vasopressor' : {'nof_classes' : 3, 'median' : 1, 'average distance' : 0.891, 'mean': 0.813},
}
TREATMENT_FEATURES = ['FiO2', 'bolus', 'vasopressor']

# WandB (Only required if logger is set to 'wandb')
# Reset me
WANDB_ARGS = {
    'project' : 'MANNERS',
    'entity' : 'MyInstitution',
}