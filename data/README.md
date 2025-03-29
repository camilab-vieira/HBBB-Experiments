# DataHandler Class 
## Overview

The `DataHandler` class is designed to simplify the process of loading, preprocessing, and preparing datasets for machine learning tasks. It supports:
- Reading datasets from multiple formats (`CSV`, `Excel`, `JSON`, `Parquet`, etc.)
- Handling both separate X and y files or combined files for features and labels.
- Splitting datasets into training and test sets.
- Imputation of missing values using different strategies.
- Encoding categorical features.
- Normalizing data using standard scaling techniques.

This class allows you to configure the preprocessing steps via a configuration file, making it flexible and easy to adapt to different datasets.

## How to Use the `DataHandler` Class

### Step 1: Create a Configuration File

Create a configuration file (`config.conf`) that specifies:
- The paths to your dataset (separate X and y files or a combined file).
- The target column for classification.
- Whether or not to apply preprocessing steps such as imputation, label encoding and normalization.


#### **Example of `.conf` file** (`data_handler.conf`)

```ini
[Drebin]
datasets = [
    {"X_path": "./data/Drebin/data_part1.csv", "y_path": "./data/Drebin/labels_part1.csv"},
    {"df_path": "./data/Drebin/data_part2.csv", "delimiter": ","}
]
target = "label"
test_size = 0.2
apply_imputation = True
imputation_strategy = "constant"
apply_label_encoding = True
encoding_method = "LabelEncoder"
apply_normalization = True
normalization_method = "MinMaxScaler"
```

### Step 2: Create the `DataHandler` Instance

In your Python script, import the `DataHandler` class and pass the configuration file path and dataset name.

```python
from data_handler import DataHandler

# Create a DataHandler instance
handler = DataHandler(config_file="data_handler.conf", dataset_name="Drebin", seed=42)

# Run the preprocessing pipeline
X_train, X_test, y_train, y_test = handler.run()
```

- `data_handler.conf` is the path to your configuration file.
- `"Drebin"` is the name of the dataset that is specified in the configuration file.
- `42` is the seed used for reproducibility (you can set any integer).

### Step 3: Access the Processed Data

After calling the `run()` method, you can access the processed training and test datasets:
- `X_train`: Preprocessed features for the training set.
- `X_test`: Preprocessed features for the test set.
- `y_train`: Target labels for the training set.
- `y_test`: Target labels for the test set.

### Step 4: View the Log File

The `DataHandler` class automatically generates a log file in the `./logs/` directory with information about the preprocessing steps applied, including the reading of datasets, splitting of data, and imputation. You can review the log file to understand how the data was processed.

### Configuration File Details

- **`datasets`**: A list of datasets, where each dataset is configured with the following parameters:
    - `X_path`: Path to the input data (features).
    - `y_path`: Path to the labels (separate from input data).
    - `df_path`: Path to a single file containing both input data and labels.
    - `delimiter`: Delimiter for the file (default is comma `,`).
- **`target`**: The name of the column containing the labels in the input data.
- **`test_size`**: Proportion of the data to be used for the test set. The default value is `0.2` (20%).
- **`apply_imputation`**: Defines whether missing value imputation should be applied. (`true` or `false`).
- **`imputation_strategy`**: The imputation strategy to use (`mean`, `median`, `most_frequent`, `constant`).
- **`imputed_value`**: Value to use for imputation when the strategy is `constant`.
- **`apply_label_encoding`**: Defines whether label encoding should be applied to categorical data. (`true` or `false`).
- **`encoding_method`**: The encoding method to use (`LabelEncoder` or `OneHotEncoder`).
- **`apply_normalization`**: Defines whether normalization should be applied to the data. (`true` or `false`).
- **`normalization_method`**: The normalization method to use (`StandardScaler` or `MinMaxScaler`).
- **`apply_Zebin_split`**: Defines whether the Zebin split should be applied, which divides the majority class into multiple parts and mixes it with the minority class.
- **`num_splits`**: Number of parts that the majority class is divide.
- **`apply_binary_label`**: If `true`, applies binary classification, mapping certain values as `0` or `1` (malignant vs benign).
- **`malignant_values`**: Specifies the values considered as malignant in the dataset when using binary labeling.
- **`benign_values`**: Specifies the values considered as benign in the dataset when using binary labeling.
- **`apply_target_encoding`**: Defines whether target encoding should be applied. (`true` or `false`).