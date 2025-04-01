import os
import sys
import shap
import json
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import matplotlib.pyplot as plt
from deslib.util.instance_hardness import kdn_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data import dataHandler
from models import modelTrainer

class DataAnalysis:
    def __init__(self, dataset_config: str, dataset_name: str, model_name: str, balance_strategy: str, balance_technique: str = 'SMOTE', num_rep=10):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.balance_strategy = balance_strategy
        self.balance_technique = balance_technique
        self.num_rep = num_rep
        self.model_folder_path = os.path.join('models/', dataset_name, balance_strategy, model_name)
        self.evaluation_folder_path = os.path.join('evaluation/', dataset_name, balance_strategy, model_name)
        os.makedirs(self.model_folder_path, exist_ok=True)
        os.makedirs(self.evaluation_folder_path, exist_ok=True)
        os.makedirs("logs/", exist_ok=True)

        self.log_file = f"/home/CIN/cbv2/HBBB-Experiments/logs/{dataset_name}_{model_name}_{balance_strategy}_{str(int(time()))}.log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')        
        
        self.handler = dataHandler.DataHandler(config_file=self.dataset_config, dataset_name=self.dataset_name)
        self.handler.read_data()
        self.handler.treat_y_df()

    
    def calculate_imbalance(self):
        class_counts = pd.Series(self.handler.y_df).value_counts(normalize=True)
        if len(class_counts) > 2:
            min_class = min(class_counts)
            max_class = max(class_counts)
            return f'1:{max_class / min_class:.2f}'
        else:
            sorted_counts = class_counts.sort_values()        
            min_class_count = sorted_counts.iloc[0]
            ratio_list = (sorted_counts / min_class_count).astype(int).astype(str)
            return ':'.join(ratio_list)

    def calculate_hardness(self):
        hardness_table = {'Unbalanced':{}, 'Balanced':{}}
        ih_scores = []

        for i in range(self.num_rep):
            handler = dataHandler.DataHandler(config_file=self.dataset_config, dataset_name=self.dataset_name, seed=i)
            handler.read_data()
            handler.treat_y_df()
            handler.split_data()
            handler.impute_missing()
            handler.encode_labels()
            x_train, y_train = handler.X_train, handler.y_train
           
            hardness_score = kdn_score(x_train, y_train, k=7)[0]
            hardness_table['Unbalanced'][f'seed-{i}'] = {}
            for label in np.unique(y_train):
                label_score = hardness_score[(y_train == label)]
                hardness_table['Unbalanced'][f'seed-{i}'][f'label_{label}'] = label_score
            logging.info(f"Hardness for seed-{i}: {np.mean(hardness_score):.4f}")

            ih_scores.append(np.mean(hardness_score))

            trainer = modelTrainer.ModelTrainer(self.dataset_config, self.dataset_name, self.model_name, self.balance_strategy)
            x_train, y_train = trainer._load_balance_techniques(x_train, y_train, 'SMOTE')
            
            hardness_score = kdn_score(x_train, y_train, k=7)[0]
            hardness_table['Balanced'][f'seed-{i}'] = {}
            for label in np.unique(y_train):
                label_score = hardness_score[(y_train == label)]
                hardness_table['Balanced'][f'seed-{i}'][f'label_{label}'] = label_score
            logging.info(f"Hardness for seed-{i}: {np.mean(hardness_score):.4f}")

            ih_scores.append(np.mean(hardness_score))

        results = {
            "mean_ih_score": float(np.mean(ih_scores)), 
            "ih_score": list(map(float, ih_scores)), 
            "hardness_table": {
                balance_type: {
                    seed: {label: [float(score) for score in scores] for label, scores in labels.items()}
                    for seed, labels in seeds.items()
                }
                for balance_type, seeds in hardness_table.items()
            }
        }

        with open(os.path.join('evaluation',f'{self.dataset_name}','results', f'{self.balance_strategy}_{self.model_name}_hardness_results.json'), "w") as f:
            json.dump(results, f, indent=4)

        return np.mean(ih_scores), hardness_table

    def plot_hardness_histograms(self, hardness_table):
        for seed in range(self.num_rep):
            plt.figure(figsize=(10, 6))
            for label in np.unique(self.handler.y_df):
                kdn_scores = hardness_table['Unbalanced'][f'seed-{str(seed)}'][f'label_{label}']
                kdn_scores = np.sort(kdn_scores)
                cumulative_score = np.arange(1, len(kdn_scores) + 1) / len(kdn_scores)
                plt.plot(kdn_scores, cumulative_score, label=f"Unbalanced-label_{label}", linestyle="-")
                kdn_scores = hardness_table['Balanced'][f'seed-{str(seed)}'][f'label_{label}']
                kdn_scores = np.sort(kdn_scores)
                cumulative_score = np.arange(1, len(kdn_scores) + 1) / len(kdn_scores)
                plt.plot(kdn_scores, cumulative_score, label=f"Balanced-label_{label}", linestyle="--")
            plt.title('Cumulative Hardness Histogram', fontsize=18)
            plt.xlabel('Hardness Score (KDN)', fontsize=16)
            plt.ylabel('Cumulative Probability', fontsize=16)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.savefig(os.path.join('evaluation',f'{self.dataset_name}','results', f'{self.balance_strategy}_{self.model_name}_{seed}_hardness.png'), dpi=300, bbox_inches='tight', pad_inches=0.2)

    def plot_shap_values(self, model, X_test):
        for file in os.listdir(self.model_folder_path):
            if file.endswith('.joblib'):
                model = joblib.load(os.path.join(self.model_folder_path, file))                
                explainer = shap.KernelExplainer(model.predict, X_test)
                shap_values = explainer.shap_values(X_test)
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(os.path.join(self.evaluation_folder_path, 'shap_summary.png'), dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    def process_jsons(self):
        json_files = [f for f in os.listdir(self.evaluation_folder_path) if f.endswith("_balance_strategy.json")]
        complete_data = []

        for file in json_files:
            file_path = os.path.join(self.evaluation_folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if key.isdigit(): 
                    complete_data.append({
                        "file": file,
                        "strategy": value["strategy"],
                        "resample_percent": value["resample_percent"]
                    })
        df = pd.DataFrame(complete_data)
        return df

    def plot_strategy_distribution(self):
        df = self.process_jsons()
        strategy_counts = df["strategy"].value_counts()
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=strategy_counts.index, y=strategy_counts.values)

        # Adicionar anotações nos gráficos
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(
                    f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold'
                )

        plt.xlabel("Strategy")
        plt.ylabel("Frequency")
        plt.title("Distribution of Balancing Strategies")
        plt.xticks(rotation=45)

        # Salvar a figura
        plt.savefig(os.path.join(self.evaluation_folder_path, 'strategies_distribution.png'), dpi=300, bbox_inches='tight', pad_inches=0.2)

    def plot_resample_histogram(self):
        df = self.process_jsons()
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df["resample_percent"], palette="Set2")
        plt.xlabel("Resampling Percentage")
        plt.ylabel("Frequency")
        plt.title("Distribution of Resampling Percentages")
        plt.savefig(os.path.join(self.evaluation_folder_path, 'resampling_distribution.png'), dpi=300, bbox_inches='tight', pad_inches=0.2)

if __name__ == "__main__":
    dataAnalysis = DataAnalysis('./config/data_handler_HBBB.conf', 'CIC-DoHBrw-2020', 'BaggingDT', 'HBBB')
    print(dataAnalysis.calculate_imbalance())
    # ih, table = dataAnalysis.calculate_hardness()
    # dataAnalysis.plot_hardness_histograms(table)
    dataAnalysis.plot_strategy_distribution()
    # dataAnalysis.plot_resample_histogram()

