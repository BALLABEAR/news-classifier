import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import joblib
import numpy as np
import torch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(self, model: Any, model_type: str, vectorizer: Optional[Any] = None,
                 vocab: Optional[Dict[str, int]] = None, max_len: Optional[int] = None,
                 device: str = "cpu"):
        self.model = model
        self.model_type = model_type
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.max_len = max_len
        self.device = device
        
        if model_type == 'neural':
            self.model.to(device)
            self.model.eval()
    
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == 'classical':
            return self._predict_classical(texts)
        else:
            return self._predict_neural(texts)
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == 'classical':
            return self._predict_proba_classical(texts)
        else:
            return self._predict_proba_neural(texts)
    
    def _predict_classical(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise ValueError("Vectorizer не найден для классической модели")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def _predict_proba_classical(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise ValueError("Vectorizer не найден для классической модели")
        
        X = self.vectorizer.transform(texts)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Для SVM используем decision_function
            decision = self.model.decision_function(X)

            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
    def _predict_neural(self, texts: List[str]) -> np.ndarray:
        if self.vocab is None:
            raise ValueError("Vocab не найден для нейронной модели")
        if self.max_len is None:
            raise ValueError("max_len не указан для нейронной модели")
        
        indices = self._texts_to_indices(texts)
        indices_tensor = torch.tensor(indices, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(indices_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def _predict_proba_neural(self, texts: List[str]) -> np.ndarray:
        if self.vocab is None:
            raise ValueError("Vocab не найден для нейронной модели")
        if self.max_len is None:
            raise ValueError("max_len не указан для нейронной модели")
        
        indices = self._texts_to_indices(texts)
        indices_tensor = torch.tensor(indices, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(indices_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    def _texts_to_indices(self, texts: List[str]) -> List[List[int]]:
        from scr.models.train_neural.utils import text_to_indices
        
        indices_list = []
        for text in texts:
            indices = text_to_indices(text, self.vocab, self.max_len)
            indices_list.append(indices)
        return indices_list


def load_classical_model(model_dir: Path, model_name: str) -> ModelWrapper:
    model_path = model_dir / f"model_{model_name}.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer не найден: {vectorizer_path}")
    
    logger.info(f"Загрузка классической модели: {model_name}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return ModelWrapper(
        model=model,
        model_type='classical',
        vectorizer=vectorizer
    )


def load_neural_model(model_dir: Path, device: str = "gpu",
                      train_neural_path: Optional[Path] = None) -> ModelWrapper:

    metadata_path = model_dir / "metadata.json"
    state_dict_path = model_dir / "model.pt"
    vocab_path = model_dir / "vocab.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json не найден: {metadata_path}")
    if not state_dict_path.exists():
        raise FileNotFoundError(f"model.pt не найден: {state_dict_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json не найден: {vocab_path}")
    
    logger.info(f"Загрузка нейронной модели из {model_dir}")
    
    # Добавляем путь к train_neural в sys.path если нужно
    if train_neural_path is not None:
        train_neural_str = str(train_neural_path)
        if train_neural_str not in sys.path:
            sys.path.insert(0, train_neural_str)
    
    # Загружаем модель используя функцию из utils
    from scr.models.train_neural.utils import load_model, load_vocab
    
    model = load_model(model_dir, device=device)
    vocab = load_vocab(vocab_path)
    
    # Получаем max_len из metadata или используем значение по умолчанию
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Пытаемся получить max_len из init_params или используем значение по умолчанию
    max_len = metadata.get('init_params', {}).get('max_len', 200)
    
    return ModelWrapper(
        model=model,
        model_type='neural',
        vocab=vocab,
        max_len=max_len,
        device=device
    )


def load_model_by_name(model_name: str, base_dir: Optional[Path] = None,
                       device: str = "cpu", train_neural_path: Optional[Path] = None) -> ModelWrapper:
    if base_dir is None:
        # Пытаемся определить базовую директорию
        current_file = Path(__file__)
        base_dir = current_file.resolve().parents[2]
    
    baseline_dir = base_dir / "models" / "baseline_models"
    neural_dir = base_dir / "models" / "neural_models"
    
    # Пробуем загрузить как классическую модель
    model_path = baseline_dir / f"model_{model_name}.pkl"
    if model_path.exists():
        logger.info(f"Найдена классическая модель: {model_name}")
        return load_classical_model(baseline_dir, model_name)
    
    # Пробуем загрузить как нейронную модель
    neural_model_dir = neural_dir / model_name
    if neural_model_dir.exists() and (neural_model_dir / "model.pt").exists():
        logger.info(f"Найдена нейронная модель: {model_name}")
        return load_neural_model(neural_model_dir, device=device, train_neural_path=train_neural_path)
    
    raise FileNotFoundError(
        f"Модель '{model_name}' не найдена ни в {baseline_dir}, ни в {neural_dir}"
    )
