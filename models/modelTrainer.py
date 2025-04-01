import os
import sys
import json
import random
import logging
import joblib
import numpy as np
import pandas as pd

from itertools import product
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from time import time

from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
)

from deslib.dcs import OLA
from deslib.des import KNOP, METADES
from deslib.static import SingleBest, StaticSelection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data import dataHandler


class ModelTrainer:
    def __init__(self, dataset_config: str, dataset_name: str, model_name: str, balance_strategy: str, balance_technique: str = 'SMOTE', n_estimators: int = 100) -> None:
        """Initialize the model trainer with logging and configuration setup."""
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.balance_strategy = balance_strategy
        self.balance_technique = balance_technique
        self.n_estimators = n_estimators
        self.model_folder_path = os.path.join('models/', dataset_name, balance_strategy, model_name)
        self.evaluation_folder_path = os.path.join('evaluation/', dataset_name, balance_strategy, model_name)
        os.makedirs(self.model_folder_path, exist_ok=True)
        os.makedirs(self.evaluation_folder_path, exist_ok=True)
        os.makedirs("logs/", exist_ok=True)

        self.log_file = f"logs/{dataset_name}_{model_name}_{balance_strategy}_{str(int(time()))}.log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')        

    def train_and_evaluate(self, num_rep: int = 10) -> None:
        """Train and evaluate the model with the specified dataset, model, and balance technique."""        
        logging.info(f"Training and evaluating model: {self.model_name} with balance strategy: {self.balance_strategy}")
        for i in tqdm(range(num_rep), desc=f"Repetition for {self.model_name} with {self.balance_strategy}"):
            handler = dataHandler.DataHandler(config_file=self.dataset_config, dataset_name=self.dataset_name, seed=i)
            X_train, X_test, y_train, y_test = handler.run()
            X_train_balanced, y_train_balanced = self._load_balance_techniques(X_train, y_train, self.balance_strategy, rep_idx=i)

            if isinstance(X_train_balanced, list):
                logging.info(f"Training ensemble: {self.model_name} with {self.balance_strategy}.")
                votes = np.zeros((len(np.array(y_test)), len(np.unique(handler.y_df))), dtype=int)
                for subset_idx in range(len(X_train_balanced)):
                    model = self._load_classifiers(X_train_balanced[subset_idx], y_train_balanced[subset_idx], self.model_name)
                    y_pred = model.predict(X_test)
                    for idx, prediction in enumerate(y_pred):
                        votes[idx, prediction] += 1
                    self.evaluate_model(y_test, y_pred, f'{i+1}_{subset_idx}')
                    
                    model_path = os.path.join(self.model_folder_path, f'model_{i+1}_{subset_idx}.joblib')
                    joblib.dump(model, model_path)
                    logging.info(f"Model {i+1}_{subset_idx} saved at {model_path}")

                logging.info(f"evaluating ensemble: {self.model_name} with {self.balance_strategy}.")
                y_pred = np.argmax(votes, axis=1)
                self.evaluate_model(y_test, y_pred, f'vote_{i+1}')

                LR = self._load_classifiers(votes, y_test, 'LogisticRegression')
                y_pred = LR.predict(votes)
                self.evaluate_model(y_test, y_pred, f'LogisticRegression_{i+1}')

                mlp = self._load_classifiers(votes, y_test, 'MLP')
                y_pred = mlp.predict(votes)
                self.evaluate_model(y_test, y_pred, f'MLP_{i+1}')
            else:
                logging.info(f"Training: {self.model_name} with {self.balance_strategy}.")
                model = self._load_classifiers(X_train_balanced, y_train_balanced, self.model_name, i)
                y_pred = model.predict(X_test)
                self.evaluate_model(y_test, y_pred, self.evaluation_folder_path, i+1)
                
                model_path = os.path.join(self.model_folder_path, f'model_{i+1}.joblib')
                joblib.dump(model, model_path)
                logging.info(f"Model {i+1} saved at {model_path}")

        # Aggregate metrics across repetitions
        logging.info(f"Evaluating across repetions: {self.model_name} with {self.balance_strategy}.")
        os.makedirs(f'evaluation/{self.dataset_name}/results', exist_ok=True)
        
        csv_path = os.path.join('evalution',f'{self.dataset_name}','results', f'{self.balance_strategy}_{self.model_name}_metrics.csv')
        self.aggregate_metrics(csv_path, self.evaluation_folder_path)
        logging.info(f'Metrics saved: {csv_path}')

    def _load_classifiers(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model_name: str, rep_idx: int = None, seed: int = 42) -> None:
        """Load and return the available classifiers."""
        classifiers = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
            'DecisionTree': DecisionTreeClassifier(criterion='gini', random_state=seed),
            'BaggingDT': BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini'), 
                                        n_estimators=self.n_estimators, n_jobs=-1, random_state=seed),
            'KNN': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            'BaggingKNN': BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=7), 
                                            n_estimators=self.n_estimators, n_jobs=-1, random_state=seed),
            'NaiveBayes': GaussianNB(),
            'BaggingNB': BaggingClassifier(estimator=GaussianNB(), 
                                        n_estimators=self.n_estimators, n_jobs=-1, random_state=seed),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed),
            'BaggingMLP': BaggingClassifier(estimator=MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000), 
                                            n_estimators=self.n_estimators, n_jobs=-1, random_state=seed),
            'GBDT': GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=seed),
            'RandomForest': RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1, random_state=seed),
            "SingleBest": lambda X_train, y_train: self._apply_ensemble(SingleBest, X_train, y_train, rep_idx),
            "StaticSelection": lambda X_train, y_train: self._apply_ensemble(StaticSelection, X_train, y_train, rep_idx),
            "OLA": lambda X_train, y_train: self._apply_ensemble(OLA, X_train, y_train, rep_idx),
            "KNOP": lambda X_train, y_train: self._apply_ensemble(KNOP, X_train, y_train, rep_idx),
            "METADES": lambda X_train, y_train: self._apply_ensemble(METADES, X_train, y_train, rep_idx),
        }
        
        return classifiers[model_name].fit(X_train, y_train)
    
    def _apply_ensemble(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame, rep_idx: int) -> None:
        logging.info(f"Training ensemble base model: {self.model_name} with {self.balance_strategy}.")
        if 'BBB' in self.balance_strategy: 
            ensemble = self.load_models_from_directory(os.path.join('models/', self.balance_strategy, 'BaggingDT'), rep_idx)
        else: 
            ensemble = self._load_classifiers(X_train, y_train, 'RandomForest', rep_idx)
        return model(ensemble).fit(X_train, y_train)

    def _load_balance_techniques(self, X_train: pd.DataFrame, y_train: pd.DataFrame, balance_technique: str, sampling_strategy = 'auto', rep_idx: int = None) -> None:
        """Load and return the available balance techniques."""
        balance = {
            'Imbalanced': self._apply_imbalanced,
            'Traditional': self._apply_traditional,
            'BBB': self._apply_bbb,
            'HBBB': lambda X, y: self._apply_hbbb(X, y, rep_idx),
            'Zebin': self._apply_zebin,

            'SMOTE': self._apply_smote,
            'RandomOverSampler': self._apply_over_sample,
            'RandomUnderSampler': self._apply_under_sample,
            'ADASYN': self._apply_adasyn,
        }

        return balance[balance_technique](X_train, y_train)
    
    def _apply_smote(self, X_train: pd.DataFrame, y_train: pd.DataFrame, sampling_strategy: str = 'auto', seed: int = 42):
        """Apply the SMOTE technique to balance the training data."""
        logging.info("Apply SMOTE.")
        smote = SMOTE(
            sampling_strategy=sampling_strategy,  
            k_neighbors=5,
            random_state=seed
        )        
        return smote.fit_resample(X_train, y_train)

    def _apply_over_sample(self, X_train: pd.DataFrame, y_train: pd.DataFrame, sampling_strategy: str = 'auto', seed: int = 42):
        """Apply the RandomOverSample technique to balance the training data."""
        logging.info("Apply RandomOverSample.")
        over_sample = RandomOverSampler(
            sampling_strategy=sampling_strategy,  
            random_state=seed
        )        
        return over_sample.fit_resample(X_train, y_train)
    
    def _apply_under_sample(self, X_train: pd.DataFrame, y_train: pd.DataFrame, sampling_strategy: str = 'auto', seed: int = 42):
        """Apply the RandomUnderSample technique to balance the training data."""
        logging.info("Apply RandomUnderSample.")
        under_sample = RandomUnderSampler(
            sampling_strategy=sampling_strategy,  
            random_state=seed
        )        
        return under_sample.fit_resample(X_train, y_train)
    
    def _apply_adasyn(self, X_train: pd.DataFrame, y_train: pd.DataFrame, sampling_strategy: str = 'auto', seed: int = 42):
        """Apply the ADASYN technique to balance the training data."""
        logging.info("Apply ADASYN.")
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,  
            random_state=seed
        )   
        return adasyn.fit_resample(X_train, y_train)
        
    def _apply_traditional(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        logging.info("Apply traditional strategy.")
        return self._load_balance_techniques(X_train, y_train, self.balance_technique)
    
    def _apply_imbalanced(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        logging.info("Apply no balance strategy.")
        return X_train, y_train  

    def _apply_zebin(self, X_train_list: pd.DataFrame, y_train_list: pd.DataFrame):
        logging.info("Apply Zebin strategy.")
        if not isinstance(X_train_list, list) or not isinstance(y_train_list, list):
            raise ValueError("Zebin requires lists of datasets.")
        balanced_subsets = [self._load_balance_techniques(X_train_list[i], y_train_list[i], self.balance_technique, rep_idx=i) for i in range(len(X_train_list))]
        X_train_balanced, y_train_balanced = zip(*balanced_subsets)
        return list(X_train_balanced), list(y_train_balanced)
        
    def _apply_bbb(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        logging.info("Apply BBB strategy.")
        balanced_subsets = []
        for j in tqdm(range(self.n_estimators)):
            X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=j)
            X_train_balanced, y_train_balanced = self._load_balance_techniques(X_resampled, y_resampled, self.balance_technique, rep_idx=j)
            balanced_subsets.append((X_train_balanced, y_train_balanced))
        X_train_balanced, y_train_balanced = zip(*balanced_subsets)
        return list(X_train_balanced), list(y_train_balanced)
    
    def _apply_hbbb(self, X_train: pd.DataFrame, y_train: pd.DataFrame, rep_idx: int):
        logging.info("Apply HBBB strategy.")
        balanced_subsets = []
        for j in tqdm(range(self.n_estimators)):
            X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=j)
            balance_technique = random.choice(['SMOTE', 'RandomOverSampler', 'RandomUnderSampler', 'ADASYN'])
            sampling_strategy = float(random.uniform(0.5, 1))

            X_train_balanced, y_train_balanced = self._load_balance_techniques(X_resampled, y_resampled, balance_technique, sampling_strategy, j)
            balanced_subsets.append((X_train_balanced, y_train_balanced))
            self.update_json(os.path.join(self.evaluation_folder_path, f'{rep_idx}__balance_strategy.json'), j, balance_technique, sampling_strategy)
        X_train_balanced, y_train_balanced = zip(*balanced_subsets)
        return list(X_train_balanced), list(y_train_balanced)
    
    def update_json(self, file_path: str, key: int, strategy: str, resample_percent):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = {}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        entries = {k: v for k, v in data.items() if k != "stats"}
        
        strategies = [entry['strategy'] for entry in entries.values()]
        resample_percents = [entry["resample_percent"] for entry in entries.values()]
        
        stats = {
            "strategy_counts": dict(Counter(strategies)),
            "resample_percent_counts": dict(Counter(resample_percents)),
        }
        data["stats"] = stats
        data[str(key)] = {"strategy": strategy, "resample_percent": resample_percent}

        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_models_from_directory(self, directory: str, rep_idx: int):
        ensemble = []
        for file in os.listdir(directory):
            if file.endswith(".joblib") and file.split('_')[-2] == rep_idx:
                model = joblib.load(os.path.join(directory, file))
                ensemble.append(model)
        return ensemble

    def evaluate_model(self, y_test: pd.DataFrame, y_pred: pd.DataFrame, classifier_id: int) -> None:
        """Evaluate model performance and save the results."""
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1_score': f1_score(y_test, y_pred, average='weighted'),
            # 'Roc_auc': roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr') if len(set(y_test)) > 2 else roc_auc_score(y_test, y_pred),
            'G-Mean': geometric_mean_score(y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Cohen_Kappa': cohen_kappa_score(y_test, y_pred),
        }
        
        # Save predictions
        df_pred = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        df_pred.to_csv(os.path.join(self.evaluation_folder_path, f'classifier_{classifier_id}_pred.csv'), index=False)

        # Save metrics
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(os.path.join(self.evaluation_folder_path, f'classifier_{classifier_id}_metrics.csv'), index=False)
        
        # Log the evaluation metrics
        logging.info(f"Evaluation metrics for classifier {classifier_id}: {metrics}")

        return metrics

    def aggregate_metrics(self, csv_path: str, model_metrics_dir: str) -> None:
        """Aggregate metrics across all repetitions and save the results."""
        metrics_list = {key: [] for key in ['Accuracy', 'Precision', 'Recall', 'F1_score', 'G-Mean', 'MCC', 'Cohen_Kappa']}
        
        for file in os.listdir(model_metrics_dir):
            if file.endswith('_metrics.csv'):
                df = pd.read_csv(os.path.join(model_metrics_dir, file))
                for metric in metrics_list.keys():
                    metrics_list[metric].append(df[metric].iloc[0])
        
        # Calculate mean and standard deviation for each metric
        aggregated_metrics = {metric: [np.mean(values), np.std(values)] for metric, values in metrics_list.items()}
        df_metrics_summary = pd.DataFrame(aggregated_metrics, index=['Mean', 'Std'])
        
        # Save aggregated metrics
        df_metrics_summary.to_csv(csv_path)
        
        logging.info(f"Aggregated metrics saved at {csv_path}")
        return df_metrics_summary


if __name__ == "__main__":
    # datasets = [('CIC-DoHBrw-2020', './config/data_handler_zebin.conf'), ('CIC-DoHBrw-2020', './config/data_handler_HBBB.conf'), ('Debrin', './config/data_handler_HBBB.conf'), ('Debrin', './config/data_handler_zebin.conf')]
    datasets = [('Drebin', './config/data_handler_zebin.conf')]
    models = ['RandomForest']
    balance_strategies = ['Zebin']

    with ProcessPoolExecutor(max_workers= 5) as executor:
        futures = [
            executor.submit(ModelTrainer(dataset_name=dataset[0], dataset_config=dataset[1],
                        model_name=model, balance_strategy=balance).train_and_evaluate, num_rep=10 )
            for dataset, model, balance in product(datasets, models, balance_strategies)
        ]
        
        for future in tqdm(futures, desc="Training and evaluating models"):
            future.result()
