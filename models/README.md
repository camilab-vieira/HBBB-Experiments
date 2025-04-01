# Model Trainer

This project provides a framework for training and evaluating machine learning models with different balancing strategies. The framework supports various classifiers, ensemble methods, and balancing techniques to handle class imbalance problems.

## Usage

### Usage in Python
```python
from model_trainer import ModelTrainer

trainer = ModelTrainer(
    dataset_name="Drebin",
    dataset_config="config/drebin_config.conf",
    model_name="RandomForest",
    balance_strategy="Traditional",
    balance_technique="SMOTE",
    n_estimators=100
)
trainer.train_and_evaluate(num_rep=10)
```

## Supported Models
- DecisionTree
- BaggingDT
- KNN
- BaggingKNN
- NaiveBayes
- BaggingNB
- MLP
- BaggingMLP
- GBDT
- RandomForest
- SingleBest
- StaticSelection
- OLA
- KNOP
- METADES

## Supported Balancing Strategies

#### 1. **Imbalanced**
- No balancing technique is applied.
- The models are trained directly on the original dataset, preserving the natural class distribution.

#### 2. **Traditional**
- A single balancing technique is applied to the entire dataset before training.
- Techniques such as **SMOTE, RandomOverSampler, RandomUnderSampler, or ADASYN** can be used.

#### 3. **BBB (Bootstrap-Based Balance)**
- Balancing is applied **within each bootstrap sample** rather than on the entire dataset.
- The BBB method **always uses SMOTE** to generate new minority class samples in each bootstrap.

#### 4. **HBBB (Hybrid Bootstrap-Based Balance)**
- Similar to BBB, this approach applies balancing **within each bootstrap sample**.
- However, instead of always using SMOTE, it **randomly selects a balancing technique** for each bootstrap (e.g., SMOTE, ADASYN, RandomOverSampler, etc.).

#### 5. **Zebin**
- The majority class is **split into multiple subgroups** to reduce its dominance.
- The **minority class is replicated** to match the number of samples within each subgroup.
- Multiple models are trained on each balanced subset in parallel.
- A **meta-model (e.g., Logistic Regression)** is used to aggregate the predictions from all models.

## Supported Balancing Techniques
- SMOTE
- RandomOverSampler
- RandomUnderSampler
- ADASYN

## Evaluation Metrics
The following metrics are computed:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- G-Mean
- MCC
- Cohen Kappa

## Logging
Logs are saved in the `logs/` directory with a timestamp-based filename.

## Model Saving
Trained models are stored in:
```
models/{balance_strategy}/{model_name}/
```

Evaluation results are stored in:
```
evaluation/{balance_strategy}/{model_name}/
```
