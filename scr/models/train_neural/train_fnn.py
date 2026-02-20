from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from utils import (
    TextDataset, load_fasttext, load_data, split_data,
    compute_metrics, save_model, save_results, logger,
    build_embedding_matrix, build_vocab, save_vocab,
)

# Конфигурация
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "processed" / "labeled_news.csv"
FASTTEXT_PATH = BASE_DIR / "data" / "embeddings" / "ruscorpora_300.bin"
MODELS_DIR = BASE_DIR / "models" / "neural_models" / "fnn"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 300
MAX_LEN = 200
HIDDEN_SIZES = [256, 128]
NUM_CLASSES = 11
DROPOUT = 0.3
USE_BATCH_NORM = True


class FNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_sizes: list,
                 num_classes: int = 11, dropout: float = 0.3, use_batch_norm: bool = True,
                 embedding_matrix: Optional[np.array] = None) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.pooling = nn.AdaptiveAvgPool1d(1)

        layers = []
        input_size = embedding_dim
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_size, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_size = hidden_size

        layers.append(nn.Linear(input_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        pooled = self.pooling(embedded).squeeze(2)
        output = self.net(pooled)
        return output


class DataLoaderUtil:
    def __init__(self, word2idx: Dict[str, int], max_len: int):
        self.word2idx = word2idx
        self.max_len = max_len

    def create_dataloaders(self, X_train: list, y_train: list,
                           X_val: list, y_val: list,
                           X_test: list, y_test: list,
                           batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset = TextDataset(X_train, y_train, self.word2idx, self.max_len)
        val_dataset = TextDataset(X_val, y_val, self.word2idx, self.max_len)
        test_dataset = TextDataset(X_test, y_test, self.word2idx, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader


class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
              patience: int = 3, min_delta: float = 1e-4) -> None:
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)

            logger.info(f"Epoch {epoch}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.04f}")

            if val_f1 - best_val_f1 > min_delta:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                logger.info(f"New best model with Val F1: {val_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with Val F1: {best_val_f1:.4f}")
        else:
            logger.warning("No improvement found, using last model")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        _, _, _, y_pred, y_true = self.validate(test_loader)
        metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
        return metrics


def main() -> None:
    logger.info("Starting FNN training pipeline")

    # 1. Загрузка данных
    X, y = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Загрузка fasttext
    word2idx = build_vocab(X_train, max_vocab_size=50000, min_freq=2)
    vocab_size = len(word2idx)
    logger.info(f"Vocabulary size: {vocab_size}")
    save_vocab(word2idx, MODELS_DIR / "vocab.json")

    fasttext = load_fasttext(FASTTEXT_PATH)
    embedding_matrix = build_embedding_matrix(fasttext, word2idx, EMBEDDING_DIM)

    # 4. Подготовка DataLoader'ов
    loader_util = DataLoaderUtil(word2idx, MAX_LEN)
    train_loader, val_loader, test_loader = loader_util.create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE
    )

    # 5. Инициализация модели
    model = FNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        use_batch_norm=USE_BATCH_NORM,
        embedding_matrix=embedding_matrix
    ).to(DEVICE)

    # 6. Обучение
    trainer = ModelTrainer(model, DEVICE, LEARNING_RATE)
    trainer.train(train_loader, val_loader, EPOCHS)

    # 7. Оценка на тесте
    metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {metrics}")

    # 8. Сохранение
    save_results(metrics, MODELS_DIR / "metrics.json")
    save_model(
        model=model,
        save_dir=MODELS_DIR,
        class_path="train_fnn.FNNClassifier",
        init_params={
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_sizes": HIDDEN_SIZES,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "use_batch_norm": USE_BATCH_NORM,
            "max_len": MAX_LEN,
        }
    )

    logger.info("FNN training completed")


if __name__ == "__main__":
    main()