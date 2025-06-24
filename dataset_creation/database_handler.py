import psycopg2
import pathlib
import gzip
import pandas as pd
from tqdm import tqdm
from typing import Optional
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, MIMIC_IV_DATA_DIR, RAW_DATA_DIR, LABEL_PATH

class DatabaseHandler:
    """
    A class to handle database operations for MIMIC-IV data, including creating databases, loading tables, and extracting specific medical data.
    This class manages PostgreSQL database operations for the MIMIC-IV critical care database, including database creation and connection management,
    table loading from CSV files, index creation for optimized querying, and data extraction for the hypotension dataset.
    """
    def __init__(self):
        self.db_name = 'mimic_iv_2_2'
        self.conn = None
        self.cursor = None
        self.tables = {
            'admissions' : {
                'csv_path' : MIMIC_IV_DATA_DIR / pathlib.Path('hosp') / pathlib.Path('admissions.csv.gz'),
                'columns' : {
                    'subject_id' : 'INTEGER NOT NULL',
                    'hadm_id' : 'INTEGER PRIMARY KEY',
                    'admittime' : 'TIMESTAMP NOT NULL',
                    'dischtime' : 'TIMESTAMP',
                    'deathtime' : 'TIMESTAMP',
                    'admission_type' : 'VARCHAR(40) NOT NULL',
                    'admit_provider_id' : 'VARCHAR(10)',
                    'admission_location' : 'VARCHAR(60)',
                    'discharge_location' : 'VARCHAR(60)',
                    'insurance' : 'VARCHAR(255)',
                    'language' : 'VARCHAR(10)',
                    'marital_status' : 'VARCHAR(30)',
                    'race': 'VARCHAR(80)',
                    'edregtime': 'TIMESTAMP',
                    'edouttime': 'TIMESTAMP',
                    'hospital_expire_flag' : 'SMALLINT'
                }
            },
            'icustays' : {
                'csv_path': MIMIC_IV_DATA_DIR / pathlib.Path('icu') / pathlib.Path('icustays.csv.gz'),
                'columns' : {
                    'subject_id' : 'INTEGER',
                    'hadm_id' : 'INTEGER REFERENCES admissions(hadm_id)',
                    'stay_id' : 'INTEGER PRIMARY KEY',
                    'first_careunit' : 'VARCHAR(50)',
                    'last_careunit' : 'VARCHAR(50)',
                    'intime' : 'TIMESTAMP(0)',
                    'outtime' : 'TIMESTAMP(0)',
                    'los' : 'DOUBLE PRECISION'
                }
            },
            'chartevents' : {
                'csv_path' : MIMIC_IV_DATA_DIR / pathlib.Path('icu') / pathlib.Path('chartevents.csv.gz'),
                'columns' : {
                    'subject_id' : 'INTEGER',
                    'hadm_id' : 'INTEGER REFERENCES admissions(hadm_id)',
                    'stay_id' : 'INTEGER REFERENCES icustays(stay_id)',
                    'caregiver_id' : 'INTEGER',
                    'charttime' : 'TIMESTAMP(0)',
                    'storetime' : 'TIMESTAMP(0)',
                    'itemid': 'INTEGER',
                    'value' : 'VARCHAR(200)',
                    'valuenum' : 'DOUBLE PRECISION',
                    'valueuom' : 'VARCHAR(20)',
                    'warning' : 'SMALLINT'
                }
            },
            'inputevents' : {
                'csv_path' : MIMIC_IV_DATA_DIR / pathlib.Path('icu') / pathlib.Path('inputevents.csv.gz'),
                'columns': {
                    'subject_id' : 'INTEGER',
                    'hadm_id' : 'INTEGER REFERENCES admissions(hadm_id)',
                    'stay_id' : 'INTEGER REFERENCES icustays(stay_id)',
                    'caregiver_id': 'INTEGER',
                    'starttime' : 'TIMESTAMP(0)',
                    'endtime' : 'TIMESTAMP(0)',
                    'storetime' : 'TIMESTAMP(0)',
                    'itemid': 'INTEGER',
                    'amount' : 'DOUBLE PRECISION',
                    'amountuom' : 'VARCHAR(30)',
                    'rate' : 'DOUBLE PRECISION',
                    'rateuom' : 'VARCHAR(30)',
                    'orderid' : 'BIGINT',
                    'linkorderid' : 'BIGINT',
                    'ordercategoryname' : 'VARCHAR(100)',
                    'secondaryordercategoryname' : 'VARCHAR(100)',
                    'ordercomponenttypedescription' : 'VARCHAR(200)',
                    'ordercategorydescription' : 'VARCHAR(50)',
                    'patientweight' : 'DOUBLE PRECISION',
                    'totalamount': 'DOUBLE PRECISION',
                    'totalamountuom': 'VARCHAR(50)',
                    'isopenbag' : 'SMALLINT',
                    'continueinnextdept' : 'SMALLINT',
                    'statusdescription' : 'VARCHAR(30)',
                    'originalamount' : 'DOUBLE PRECISION',
                    'originalrate' : 'DOUBLE PRECISION'
                }
            },
            'outputevents' : {
                'csv_path': MIMIC_IV_DATA_DIR / pathlib.Path('icu') / pathlib.Path('outputevents.csv.gz'),
                'columns' : {
                    'subject_id' : 'INTEGER',
                    'hadm_id' : 'INTEGER REFERENCES admissions(hadm_id)',
                    'stay_id' : 'INTEGER REFERENCES icustays(stay_id)',
                    'caregiver_id': 'INTEGER',
                    'charttime' : 'TIMESTAMP(3)',
                    'storetime' : 'TIMESTAMP(3)',
                    'itemid': 'INTEGER',
                    'value' : 'DOUBLE PRECISION',
                    'valueuom' : 'VARCHAR(20)'
                }
            }
        }
        self.cohort_table_name = 'hypotension_stays'

    def __enter__(self):
        self.conn = self.create_db_if_not_exists()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cursor.close()
        self.conn.close()
        pass

    def db_exists(self, cursor: Optional[psycopg2.extensions.cursor] = None):
        if cursor is None:
            conn = psycopg2.connect(database='postgres', user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
            cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{self.db_name}'")
        exists = cursor.fetchone()
        return exists is not None

    def create_db_if_not_exists(self):
        conn = psycopg2.connect(database='postgres', user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
        conn.autocommit = True
        cursor = conn.cursor()
        exists = self.db_exists(cursor)
        # cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{self.db_name}'")
        # exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {self.db_name}")
            print('Created database', self.db_name)
        return psycopg2.connect(database=self.db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
    
    def load_table(self, table_name: str, csv_path: pathlib.Path, columns: dict, overwrite: bool = False):
        if overwrite:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        print(f"Loading {table_name} from {csv_path}")
        with gzip.open(csv_path, 'rt') as file:
            header = next(file).strip().split(',') # skip header
            column_names = list(columns.keys())
            if len(header) != len(column_names):
                raise ValueError("Column names and the header of the CSV file do not have the same length")
            for idx, header_entry in enumerate(header):
                if header_entry != column_names[idx]:
                    raise ValueError("Column names and the header of the CSV file differ at index " + str(idx) + ', header: "' + header_entry + '", columns :"' + column_names[idx] + '"')
            self.cursor.execute(f"CREATE TABLE {table_name} ({', '.join([f'{col} {data_type}' for col, data_type in columns.items()])})")
            print('Inserting data')
            copy_statement = f"COPY {table_name} FROM STDIN DELIMITER ',' NULL '' ESCAPE '\\' CSV"
            self.cursor.copy_expert(copy_statement, file)
            print('Created table', table_name)

    def load_into_db(self):
        self.create_db_if_not_exists()
        for table, table_data in self.tables.items():
            table_path = table_data['csv_path']
            columns = table_data['columns']
            self.load_table(table, table_path, columns, overwrite=True)

    def create_indices(self):
        to_index = {
            'admissions' : ['hadm_id'],
            'icustays' : ['stay_id', 'hadm_id'],
            'chartevents' : ['stay_id', 'itemid', 'charttime'],
            'inputevents' : ['stay_id', 'itemid', 'starttime'],
            'outputevents' : ['stay_id', 'itemid', 'charttime'],
        }

        statements = []

        for table, vars_to_index in to_index.items():
            for var in vars_to_index:
                statements.append(f"DROP INDEX IF EXISTS idx_{table}_{var}")
                statements.append(f"CREATE INDEX idx_{table}_{var} ON {table}({var})")
        print('Creating indices')
        for statement in tqdm(statements):
            self.cursor.execute(statement)

    def query_cohort(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.cohort_table_name}")
        self.cursor.execute(f"CREATE TABLE {self.cohort_table_name} (stay_id INTEGER PRIMARY KEY, hadm_id INTEGER, subject_id INTEGER, icu_intime TIMESTAMP(0), expire_flag SMALLINT)")
        print('Creating cohort table')

        cohort_query = f"""
        WITH low_maps AS (
            SELECT p.stay_id, itemid
            FROM chartevents p
            INNER JOIN icustays q ON p.stay_id = q.stay_id
            WHERE itemid IN (220052, 220181, 225312)
                AND valuenum <= 65
                AND valuenum > 0
                AND extract(epoch FROM (charttime - intime))/3600 <= 48
        ), ht_stays AS (
            SELECT p.subject_id, p.hadm_id, p.stay_id, intime, hospital_expire_flag, COUNT(itemid) AS num_low_maps
            FROM icustays p
            RIGHT JOIN low_maps q ON p.stay_id = q.stay_id
            LEFT JOIN admissions k on p.hadm_id = k.hadm_id
            GROUP BY p.subject_id, p.hadm_id, p.stay_id, intime, hospital_expire_flag
            HAVING COUNT(itemid) >= 7
        )
        SELECT stay_id, hadm_id, subject_id, intime, hospital_expire_flag
        FROM ht_stays
        ORDER BY stay_id
        """
        print('Retrieving cohort data')
        self.cursor.execute(cohort_query)
        cohort_data = self.cursor.fetchall()
        print('Got ' + str(len(cohort_data)) + ' stays in cohort')       

        insert_query = f"""
        INSERT INTO {self.cohort_table_name} (stay_id, hadm_id, subject_id, icu_intime, expire_flag)
        VALUES (%s, %s, %s, %s, %s)
        """
        print('Inserting into table')
        self.cursor.executemany(insert_query, cohort_data)
        print('Storing in CSV')
        df = pd.DataFrame(cohort_data, columns=['stay_id', 'hadm_id', 'subject_id', 'icu_intime', 'survival'])
        df['survival'] = 1 - df['survival']
        df.to_csv(LABEL_PATH, index=False)

    def extract_charted_data(self):
        chart_item_dict = {
            220615 : 'serum_creatinine', # mg/dL
            223835 : 'FiO2', # no unit
            225668 : 'lactate', # mmol/L
            220644 : 'ALT', # IU/L
            220587 : 'AST', # IU/L
            220050 : 'SBP', # mmHg
            220179 : 'SBP', # mmHg
            224167 : 'SBP', # mmHg
            225309 : 'SBP', # mmHg
            227243 : 'SBP', # mmHg
            220051 : 'DBP', # mmHg
            220180 : 'DBP', # mmHg
            224643 : 'DBP', # mmHg
            225310 : 'DBP', # mmHg
            227242 : 'DBP', # mmHg
            220052 : 'MAP', # mmHg
            220181 : 'MAP', # mmHg
            225312 : 'MAP', # mmHg
            220224 : 'pO2', # mmHg
            220739 : 'GCS_eyes', # no unit
            223900 : 'GCS_verbal', # no unit
            223901 : 'GCS_motor' # no unit
        }
        charted_query = f"""
        SELECT p.stay_id, itemid, charttime, valuenum AS value
        FROM chartevents p
        INNER JOIN {self.cohort_table_name} q ON p.stay_id = q.stay_id
        WHERE ({' OR '.join([f'itemid = {item}' for item in chart_item_dict.keys()])}) AND valuenum > 0 AND extract(epoch FROM (charttime - icu_intime))/3600 <= 48
        """
        print('Query charted data')
        self.cursor.execute(charted_query)
        chart_data = self.cursor.fetchall()
        print('Got ' + str(len(chart_data)) + ' charted data rows')
        print('Store in CSV')
        df = pd.DataFrame(chart_data, columns=['stay_id', 'item', 'charttime', 'value'])
        df['value'] = df['value'].astype(float)
        df['item'] = df['item'].map(chart_item_dict)
        df.to_csv(RAW_DATA_DIR / pathlib.Path('charted_data.csv'), index=False)

    def extract_bolus_data(self):
        bolus_item_dict = {
            225158 : 'NaCl',
            220955 : 'Ringers_lactate',
            225168 : 'Packed_RBC',
            220970 : 'Frozen_plasma',
            225170 : 'Platelets'
        }
        bolus_query = f"""
        SELECT p.stay_id, itemid, starttime, amount AS value
        FROM inputevents p
        INNER JOIN {self.cohort_table_name} q ON p.stay_id = q.stay_id
        WHERE ({' OR '.join([f'itemid = {item}' for item in bolus_item_dict.keys()])})
            AND amount >= 250
            AND extract(epoch FROM (starttime - icu_intime))/3600 <= 48
            AND extract(epoch FROM (endtime - starttime)) = 60
        """
        print('Query for bolus data')
        self.cursor.execute(bolus_query)
        bolus_data = self.cursor.fetchall()
        print('Got ' + str(len(bolus_data)) + ' bolus data rows')
        print('Store in CSV')
        df = pd.DataFrame(bolus_data, columns=['stay_id', 'item', 'charttime', 'value'])
        df['value'] = df['value'].astype(float)
        df['item'] = df['item'].map(bolus_item_dict)
        df.to_csv(RAW_DATA_DIR / pathlib.Path('bolus_data.csv'), index=False)

    def __adjust_vasopressor_rate(self, row, factor_dict):
        item = row['item']
        result = row['rate'] * factor_dict[item]
        if item == 'Norepinephrine':
            if row['unit'] == 'mg/kg/min':
                result *= 1000
        elif item == 'Vasopressin':
            if row['unit'] == 'units/hour':
                result /= 60
        elif item == 'Phenylephrine':
            if row['unit'] == 'mcg/min':
                result /= row['weight']
        return result

    def extract_vasopressor_data(self):
        vasopressor_item_dict = {
            221906 : 'Norepinephrine', # mcg/kg/min or mg/kg/min -> mcg/kg/min
            222315 : 'Vasopressin', # units/hour, units/min -> units/min
            221749 : 'Phenylephrine', # mcg/kg/min, mcg/min -> mcg/kg/min
            221662 : 'Dopamine', # mcg/kg/min
            221289 : 'Epinephrine' # mcg/kg/min
        }
        factor_dict = {
            'Norepinephrine' : 1.0,
            'Vasopressin' : 5.0,
            'Phenylephrine' : 0.45,
            'Dopamine' : 0.01,
            'Epinephrine' : 1.0
        }
        vasopressor_query = f"""
        SELECT p.stay_id, itemid, starttime, endtime, rate, rateuom as unit, patientweight
        FROM inputevents p
        INNER JOIN {self.cohort_table_name} q ON p.stay_id = q.stay_id
        WHERE ({' OR '.join([f'itemid = {item}' for item in vasopressor_item_dict.keys()])})
            AND rate > 0
            AND extract(epoch FROM (starttime - icu_intime))/3600 <= 48
        """
        print('Query for vasopressor data')
        self.cursor.execute(vasopressor_query)
        vasopressor_data = self.cursor.fetchall()
        print('Got ' + str(len(vasopressor_data)) + ' vasopressor data rows')
        print('Calculate adjusted rates')
        df = pd.DataFrame(vasopressor_data, columns=['stay_id', 'item', 'starttime', 'endtime', 'rate', 'unit', 'weight'])
        df['item'] = df['item'].map(vasopressor_item_dict)
        df['adjusted_rate'] = df.apply(lambda x: self.__adjust_vasopressor_rate(x, factor_dict), axis=1)
        print('Store in CSV')
        df.to_csv(RAW_DATA_DIR / pathlib.Path('vasopressor_data.csv'), index=False)

    def extract_outputevents_data(self):
        output_ids = [226566, 226627, 226631, 226559, 226561, 226567, 226632, 226557, 226558, 226563]
        output_query = f"""
        SELECT p.stay_id, charttime, value
        FROM outputevents p
        INNER JOIN {self.cohort_table_name} q ON p.stay_id = q.stay_id
        WHERE ({' OR '.join([f'itemid = {item}' for item in output_ids])}) AND value > 0 AND extract(epoch FROM (charttime - icu_intime))/3600 <= 48
        """
        print('Query output events data')
        self.cursor.execute(output_query)
        output_data = self.cursor.fetchall()
        print('Got ' + str(len(output_data)) + ' output event data rows')
        df = pd.DataFrame(output_data, columns=['stay_id', 'charttime', 'value'])
        df['item'] = 'urine'
        print('Store in CSV')
        df.to_csv(RAW_DATA_DIR / pathlib.Path('outputevents_data.csv'), index=False)

    def load_data(self):
        self.load_into_db()
        self.create_indices()

    def extract_data(self):
        self.query_cohort()
        self.extract_charted_data()
        self.extract_bolus_data()
        self.extract_vasopressor_data()
        self.extract_outputevents_data()
        self.add_48h_survival_and_los()

    def add_48h_survival_and_los(self):
        self.cursor.execute(f"""
        ALTER TABLE {self.cohort_table_name}
        ADD COLUMN los DOUBLE PRECISION,
        ADD COLUMN icu_outtime TIMESTAMP(0),
        ADD COLUMN hosp_discharge_time TIMESTAMP,
        ADD COLUMN hosp_death_time TIMESTAMP
        """)
        print('Adding LOS and discharge columns')
        self.cursor.execute(f"""
        UPDATE {self.cohort_table_name} p
        SET los = q.los, icu_outtime = q.outtime
        FROM (
            SELECT p.stay_id, p.los, outtime
            FROM icustays p
            INNER JOIN {self.cohort_table_name} q ON p.stay_id = q.stay_id
        ) q
        WHERE p.stay_id = q.stay_id
        """)
        self.cursor.execute(f"""
        UPDATE {self.cohort_table_name} p
        SET hosp_discharge_time = q.dischtime, hosp_death_time = q.deathtime
        FROM (
            SELECT p.hadm_id, dischtime, deathtime
            FROM admissions p
            INNER JOIN {self.cohort_table_name} q ON p.hadm_id = q.hadm_id
        ) q
        WHERE p.hadm_id = q.hadm_id
        """)
        print('Storing in CSV')
        self.cursor.execute(f"SELECT * FROM {self.cohort_table_name} ORDER BY stay_id")
        cohort_data = self.cursor.fetchall()
        df = pd.DataFrame(cohort_data, columns=['stay_id', 'hadm_id', 'subject_id', 'icu_intime', 'survival', 'los', 'icu_outtime', 'hosp_discharge_time', 'hosp_death_time'])
        df['survival'] = 1 - df['survival']
        df.to_csv(LABEL_PATH, index=False)