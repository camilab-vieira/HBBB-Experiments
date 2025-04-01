import os
import sys
import logging
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import utils

class DataHandler:
    def __init__(self, config_file: str, dataset_name: str, seed: int = None) -> None:
        self.config = utils.load_config(config_file, dataset_name)
        self.seed = seed if seed else int(time())
        
        os.makedirs("logs/", exist_ok=True)
        self.log_file = f"logs/{config_file.split('/')[-1].split('.')[0]}_{dataset_name}_{self.seed}_{str(int(time()))}.log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.X_df = None
        self.y_df = None
        self.encoders = {}
        self.scaler = None
        self.imputer = None
        self.label_encoder = LabelEncoder()

    def read_data(self) -> None:
        """Read data from file(s)"""
        datasets = self.config["datasets"]
        X_list = []
        y_list = []
        
        for dataset in datasets:
            # Check if we are using separate files for X and y
            X_path = dataset.get("X_path")
            df_path = dataset.get("df_path")
            y_path = dataset.get("y_path")
            delimiter = dataset.get("delimiter", ",")

            if not (X_path or df_path):
                raise ValueError("Either X_path or df_path must be provided for each dataset.")
            
            if X_path and y_path:
                # Separate files for X and y
                X_df = self._read_file(X_path, delimiter)
                y_df = self._read_file(y_path, delimiter)
            elif X_path or df_path:
                if X_path:
                    df_path = X_path 
                # Combined file for X and y
                X_df = self._read_file(df_path, delimiter)
                target_col = self.config["target"]
                if target_col in X_df.columns:
                    y_df = X_df[target_col]
                    X_df = X_df.drop(columns=[target_col])
                else:
                    raise ValueError(f"Target column '{target_col}' not found in the dataset.")
            else:
                raise ValueError("X_path must be provided in the configuration.")

            # Ensure that X and y are aligned
            if X_df.shape[0] != y_df.shape[0]:
                raise ValueError("The number of rows in X and y must match.")
            
            logging.info(f"Data reading completed. X shape: {X_df.shape}, y shape: {y_df.shape}")
            X_list.append(X_df)
            y_list.append(y_df)
        
        # Concatenate all dataframes in X_list and y_list
        self.X_df = pd.concat(X_list, ignore_index=True)
        self.y_df = pd.concat(y_list, ignore_index=True)

        logging.info(f"Data reading completed. X shape: {self.X_df.shape}, y shape: {self.y_df.shape}")

    def _read_file(self, path: str, delimiter: str) -> None:
        """Helper function to read a file based on its extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.csv', '.txt']:
            return pd.read_csv(path, delimiter=delimiter)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif ext == '.json':
            return pd.read_json(path)
        elif ext == '.parquet':
            return pd.read_parquet(path)
        elif ext == '.npy':
            return pd.DataFrame(np.load(path))
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def treat_y_df(self) -> None:
        self.y_df = self.y_df.squeeze()
        if self.config.get("apply_binary_label", False):
            malignant_values = self.config.get("malignant_values")
            benign_values = self.config.get("benign_values")
            if malignant_values:
                self.y_df = self.y_df.apply(lambda x: 1 if x in malignant_values else 0)
            if benign_values:
                self.y_df = self.y_df.apply(lambda x: 0 if x in benign_values else 1)
            logging.info("Target binarization applied: 0 represents benign samples, while 1 represents malignant samples.")

        if self.config.get("apply_target_encoding", False):
            self.y_df = self.label_encoder.fit_transform(self.y_df)
            class_mapping = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
            logging.info(f"Target encoding: {class_mapping}")
    
    def split_data(self):
        test_size = self.config.get("test_size", 0.2)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_df, self.y_df, test_size=test_size, random_state=self.seed, stratify=self.y_df
        )
        logging.info(f"Data split. Training size: {self.X_train.shape}, Test size: {self.X_test.shape}")
    
    
    def encode_labels(self) -> None:
        categorical_cols = self.X_train.select_dtypes(include=["object"]).columns
        method = self.config.get("encoding_method", "LabelEncoder")
        
        if method == "OneHotEncoder":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.X_train = pd.DataFrame(encoder.fit_transform(self.X_train[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
            self.X_test = pd.DataFrame(encoder.transform(self.X_test[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
            logging.info(f"OneHotEncoding applied. Number of columns after encoding: {self.X_train.shape[1]}")
        elif method == "LabelEncoder":
            for col in categorical_cols:
                encoder = LabelEncoder()
                self.X_train[col] = encoder.fit_transform(self.X_train[col])
                self.X_test[col] = encoder.transform(self.X_test[col])
                self.encoders[col] = encoder
            logging.info(f"LabelEncoding applied. Categorical columns encoded: {categorical_cols}")
        elif method == "Drop":
            self.X_train.drop(columns=categorical_cols, axis=1, inplace=True)
            self.X_test.drop(columns=categorical_cols, axis=1, inplace=True)
            logging.info(f"Drop categorical columns applied. Categorical columns: {categorical_cols}")
    
    def impute_missing(self) -> None:
        strategy = self.config.get("imputation_strategy", "constant")
        for col in self.X_train.columns:
            if self.X_train[col].isnull().any():
                missing_count = self.X_train[col].isnull().sum()
                if "imputed_value" in self.config:
                    imputed_value = self.config["imputed_value"]
                elif strategy == "mean":
                    imputed_value = self.X_train[col].mean()
                elif strategy == "median":
                    imputed_value = self.X_train[col].median()
                elif strategy == "most_frequent":
                    imputed_value = self.X_train[col].mode()[0]
                else:  # Default or "constant"
                    min_value = self.X_train[col].min()
                    imputed_value = min_value - 1 if not np.isnan(min_value) else -1
                
                self.X_train[col].fillna(imputed_value, inplace=True)
                self.X_test[col].fillna(imputed_value, inplace=True)
                logging.info(f"Imputed {missing_count} missing values in column '{col}' with value {imputed_value}")
    
    def normalize(self) -> None:
        method = self.config.get("normalization_method", "StandardScaler")
        if method == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)
        logging.info(f"Normalization applied with {method}. Shape after normalization: {self.X_train.shape}")
    
    def zebin_split(self) -> None:
        self.X_train = pd.DataFrame(self.X_train).reset_index(drop=True)
        self.y_train = pd.Series(self.y_train).reset_index(drop=True)

        num_splits = self.config.get("num_splits", 3)

        # Identify the majority class
        class_counts = self.y_train.value_counts()
        majority_class = class_counts.idxmax()

        # Separate majority class data from minority class data
        majority_X = self.X_train[self.y_train == majority_class]
        majority_y = self.y_train[self.y_train == majority_class]
        minority_X = self.X_train[self.y_train != majority_class]
        minority_y = self.y_train[self.y_train != majority_class]

        majority_X_splits = np.array_split(majority_X, num_splits)
        majority_y_splits = np.array_split(majority_y, num_splits)

        self.X_train_splits = []
        self.y_train_splits = []

        for i in range(num_splits):
            X_train = pd.concat([majority_X_splits[i], minority_X])
            y_train = pd.concat([pd.Series(majority_y_splits[i]), pd.Series(minority_y)])

            self.X_train_splits.append(X_train)
            self.y_train_splits.append(y_train)
            logging.info(f"Zebin split {i} applied: {X_train.shape}.")

        self.X_train = self.X_train_splits
        self.y_train = self.y_train_splits
        logging.info(f"Zebin split applied with {num_splits} subsets. Each subset includes all minority classes.")

    def run(self) -> None:
        self.read_data()
        self.treat_y_df()
        self.split_data()
        if self.config.get("apply_imputation", False):
            self.impute_missing()
        if self.config.get("apply_label_encoding", False):
            self.encode_labels()
        if self.config.get("apply_normalization", False):
            self.normalize()
        if self.config.get("apply_zebin_split", False):
            self.zebin_split()
        
        logging.info("Preprocessing completed successfully.")
        return self.X_train, self.X_test, self.y_train, self.y_test
    