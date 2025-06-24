from database_handler import DatabaseHandler
from dataset_preparator import DataPreparator

"""A script to create and preprocess the hypotension dataset of 
Gottesman et al. (https://doi.org/10.48550/arXiv.2002.03478) extracted from the MIMIC-IV database. 
"""

# Set to True to overwrite existing data
OVERWRITE = False

if __name__ == "__main__":
    with DatabaseHandler() as db_handler:
        if not OVERWRITE and db_handler.db_exists():
            print('Database already exists. Skipping data loading.')
        else:
            print('STEP 1/3: Load data into database')
            db_handler.load_data()
            print('STEP 2/3: Extract relevant data from database')
            db_handler.extract_data()
    if not OVERWRITE and DataPreparator.prepared_data_exists():
        print('Dataset already exists. Skipping dataset preparation.')
    else:
        print('STEP 3/3: Prepare dataset')
        preparator = DataPreparator()
        preparator.prepare_dataset()