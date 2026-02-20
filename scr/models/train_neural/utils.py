import logging
import json
import importlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from gensim.models import KeyedVectors

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], max_len: int):
        self.indices = []
        self.labels = labels

        for text in texts:
            idx_seq = text_to_indices(text, word2idx, max_len)
            self.indices.append(idx_seq)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_seq = self.indices[idx]
        label = self.labels[idx]

        idx_tensor = torch.tensor(idx_seq, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return idx_tensor, label_tensor


def build_vocab(texts: List[str], max_vocab_size: Optional[int] = None, min_freq: int = 1) -> Dict[str, int]:
    all_words = []
    for text in texts:
        all_words.extend(text.split())

    word_counts = Counter(all_words)

    vocab_words = [word for word, count in word_counts.items() if count >= min_freq]

    vocab_words.sort(key=lambda word: word_counts[word], reverse=True)

    if max_vocab_size is not None:
        vocab_words = vocab_words[:max_vocab_size]

    word2idx = {'<PAD>':0, '<UNK>': 1}
    for idx, word in enumerate(vocab_words, start=2):
        word2idx[word] = idx

    logger.info(f"Vocabulary built: {len(word2idx)} words (min_freq={min_freq}, max_vocab_size={max_vocab_size})")
    return word2idx


def text_to_indices(text: str, word2idx: Dict[str, int], max_len: int) -> List[int]:
    words = text.split()
    unk_idx = word2idx.get('<UNK>', 1)

    indices = [word2idx.get(word, unk_idx) for word in words]

    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        pad_len = max_len - len(indices)
        indices.extend([word2idx['<PAD>']] * pad_len)

    return indices


def load_fasttext(path: Path) -> KeyedVectors:
    if not path.exists():
        raise FileNotFoundError(f"Файл fasttext не найден: {path}")

    try:
        if path.suffix == '.bin':
            model = KeyedVectors.load_word2vec_format(str(path), binary=True)
        else:
            model = KeyedVectors.load_word2vec_format(str(path), binary=False)
    except Exception as e:
        logger.error(f"Ошибка загрузки fasttext из {path}: {e}")
        raise

    logger.info(f"Fasttext модель загружена. Размерность: {model.vector_size}, слов: {len(model.key_to_index)}")
    return model


def build_embedding_matrix(fasttext_model: KeyedVectors, word_index: Dict[str, int], vector_size: int = 300) -> np.ndarray:
    vocab_size = len(word_index)

    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, vector_size)).astype(np.float32)

    for word, idx in word_index.items():
        if word in fasttext_model.key_to_index:
            embedding_matrix[idx] = fasttext_model[word]

    logger.info(f"Матрица эмбеддингов создана: {vocab_size} x {vector_size}")
    return embedding_matrix


def load_data(data_path: Path) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(data_path)
    if 'lemmatized_text' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV должен содержать колонки "lemmatized_text" и "label"')
    texts = df['lemmatized_text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    logger.info(f"Загружено {len(texts)} примеров")
    return texts, labels


def split_data(X: List[str], y: List[int], test_size: float = 0.2,
               val_size: float = 0.1, random_state: int = 42) -> Tuple:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_size_adj = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adj, random_state=random_state, stratify=y_train_val
    )

    logger.info(f"Разделение: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    logger.info(f"Метрики вычислены: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def save_model(model: nn.Module, save_dir: Path, class_path: str, init_params: Dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")

    metadata = {
        "class_path": class_path,
        "init_params": init_params,
    }
    with open(save_dir / "metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Модель сохранена в {save_dir}")


def save_results(metrics: Dict[str, float], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Метрики сохранены в {path}")


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Словарь сохранён в {path}, размер: {len(vocab)}")


def load_model(model_dir: Path, device: str = "cpu") -> nn.Module:
    metadata_path = model_dir / "metadata.json"
    state_dict_path = model_dir / "model.pt"

    if not metadata_path.exists() or not state_dict_path.exists():
        raise FileNotFoundError(f"Файлы модели не найдены в {model_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_path = metadata["class_path"]
    init_params = metadata["init_params"]

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model = model_class(**init_params)

    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Модель загружена из {model_dir}")
    return model


def load_vocab(path: Path) -> Dict[str, int]:
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    logger.info(f"Словарь загружен из {path}, размер: {len(vocab)}")
    return vocab