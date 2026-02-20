import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.processing_config import CATEGORIES_MAPPING, CATEGORIES_TO_REMOVE, CATEGORY_RULES

# настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# конфигурация
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "news_clean.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "labeled_news.csv"

# инициализация глобальных объектов
def apply_category_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    df = df.copy()

    for target_category, keywords in rules.items():
        mask = df['lemmatized_text'].str.contains(
            '|'.join(keywords),
            case=False,
            na=False,
            regex=True
        )

        df.loc[mask, 'category_clean'] = target_category

        logger.info(f"Применены правила для '{target_category}': изменено {mask.sum()} строк")

    return df

def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['category_clean'] = (
        df['category']
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )

    df = df[~df['category_clean'].isin(CATEGORIES_TO_REMOVE)]
    logger.info(f"{'После удаления categories_to_remove':<40} | rows: {len(df)}")

    df['category_clean'] = df['category_clean'].map(CATEGORIES_MAPPING)
    df = df.dropna(subset=['category_clean'])

    df = apply_category_rules(df, CATEGORY_RULES)

    unique_categories = df['category_clean'].unique()
    label_map = {cat: i for i, cat in enumerate(unique_categories)}
    df['label'] = df['category_clean'].map(label_map)

    df = df.drop_duplicates(subset=['lemmatized_text'], keep='first')
    logger.info(f"{'После удаления дубликатов':<40} | rows: {len(df)}")

    return df

def save_data(df: pd.DataFrame, path: Path) -> bool:
    try:
        df[['lemmatized_text', 'category_clean', 'label']].to_csv(OUTPUT_PATH, index=False)
        logger.info(f'Data saved at {path}')
        return True
    except Exception as e:
        logger.error(f'Failed to save data: {e}')
        return False

def plot_data(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x='category_clean', order=df['category_clean'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Распределение категорий')
    plt.show()

def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"{'Загружен исходный датасет':<40} | rows: {len(df)}")

    result_df = clean_and_label(df)
    success = save_data(result_df, OUTPUT_PATH)

    if success:
        logger.info("Preprocessing completed successfully.")
    else:
        logger.error("Preprocessing failed during saving.")

    plot_data(result_df)


if __name__ == "__main__":
    main()