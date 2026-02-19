import pandas as pd
import joblib

import json
import logging
from tqdm import tqdm
from typing import Dict, Any, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)
import lightgbm as lgb

# настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# конфигурация
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "labeled_news.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR = MODELS_DIR / "baseline_models"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# классы и функции
class DataLoader:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

        return self.df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        X = df['lemmatized_text'].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=True
        )

        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_models(self) -> Dict[str, Pipeline]:
        models = {
            "lr": LogisticRegression(random_state=self.random_state, solver="saga", max_iter=2000),
            "svm": LinearSVC(random_state=self.random_state, dual=False, max_iter=2000),
            "lgbm": lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        }
        return models

    def get_param_grids(self) -> Dict[str, Dict[str, Any]]:
        param_grids = {}

        param_grids['lr'] = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        }
        param_grids['svm'] = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            'class_weight': [None, 'balanced']
        }
        param_grids['lgbm'] = {
            'n_estimators': [50],
            'learning_rate': [0.1],
            'num_leaves': [31],
            'max_depth': [10],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'max_bin': [63]
        }

        return param_grids

    def tune_model(self, model_name: str, estimator, param_dist: Dict,
                   X_train_vec, y_train) -> Tuple[Any, Dict[str, Any]]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        search = GridSearchCV(
            estimator,
            param_grid=param_dist,
            cv=cv,
            scoring='f1_macro',
            verbose=2,
            n_jobs=-1
        )

        search.fit(X_train_vec, y_train)
        logger.info(f"Best parameters for {model_name}: {search.best_params_}  best_score={search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        return metrics


class ModelSaver:
    @staticmethod
    def save_model(model, path: Path):
        joblib.dump(model, path)

    @staticmethod
    def save_results(metrics: Dict, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)


# основной пайплайн
def main():
    logger.info("Starting baseline training pipeline")

    # 1. Load and split data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(df)

    # 2. TF-IDF
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        max_df=0.7,
        min_df=3
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    joblib.dump(tfidf, TEMP_DIR / "tfidf_vectorizer.pkl", compress=3)
    logger.info("Tfidf vectorizer saved.")

    # 2. Initialize trainer
    trainer = ModelTrainer(random_state=42)
    models = trainer.get_models()
    param_grids = trainer.get_param_grids()

    # 3. Train and tune each model
    best_models = {}
    results = {}

    pbar = tqdm(total=len(models), desc="Training models")
    for name, estimator in models.items():
        logger.info(f"Processing model: {name}")
        best_model, best_params = trainer.tune_model(
            name, estimator, param_grids[name],
            X_train_vec, y_train
        )
        best_models[name] = best_model
        metrics = trainer.evaluate(best_model, X_test_vec, y_test)
        results[name] = metrics

        with open(TEMP_DIR / f"results_{name}.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        joblib.dump(best_model, TEMP_DIR / f"model_{name}.pkl", compress=3)
        with open(TEMP_DIR / f"params_{name}.json", 'w') as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Results for {name}: {metrics}")
        pbar.update(1)
    pbar.close()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()