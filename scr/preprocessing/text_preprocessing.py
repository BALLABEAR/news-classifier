import pandas as pd
from tqdm import tqdm
from typing import  Optional

from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import nltk
import logging

from scr import db
from config.processing_config import EXTRA_STOPWORDS, CLEANING_PATTERNS
from pathlib import Path

# настройка логирования
logger = logging.getLogger(__name__)

# конфигурация
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "news_clean.csv"

# инициализация глобальных объектов
def load_data(query: str) -> Optional[pd.DataFrame]:
    try:
        conn = db.get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        logger.info(f'Loaded {len(df)} articles')
        return df
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        return None

def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.str.replace(CLEANING_PATTERNS['tags'], '', regex=True)
    cleaned = cleaned.str.replace(CLEANING_PATTERNS['urls'], '', regex=True)
    cleaned = cleaned.str.replace(CLEANING_PATTERNS['non_alpha'], '', regex=True)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(CLEANING_PATTERNS['extra_spaces'], ' ', regex=True).str.strip()

    logger.info(f'Cleaned completed')
    return cleaned

def get_stopwords():
    base_stopwords = set(stopwords.words('russian'))
    return base_stopwords.union(EXTRA_STOPWORDS)

def remove_stopwords_from_series(series: pd.Series, stopwords_set: set) -> pd.Series:
    tqdm.pandas(desc='Stopwords removal')
    return series.progress_apply(
        lambda text: ' '.join([word for word in text.split() if word not in stopwords_set]),
    )

def remove_stopwords_from_tokens(series: pd.Series, stopwords_set: set) -> pd.Series:
    tqdm.pandas(desc='Stopwords removal from tokens')
    return series.progress_apply(
        lambda tokens: [word for word in tokens if word not in stopwords_set]
    )

def tokenize_series(series: pd.Series) -> pd.Series:
    tqdm.pandas(desc='Tokenization')
    return series.progress_apply(
        lambda text: nltk.word_tokenize(text, language='russian')
    )

def lemmatize_series(series: pd.Series) -> pd.Series:
    morph = MorphAnalyzer()
    tqdm.pandas(desc='Lemmatization')
    return series.progress_apply(
        lambda token: [morph.parse(t)[0].normal_form for t in token]
    )

def lemmas_to_string_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda lemmas: ' '.join(lemmas))

def save_data(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_csv(path, index=False, encoding='utf-8')
        logger.info(f'Data saved at {path}')
        return True
    except Exception as e:
        logger.error(f'Failed to save data: {e}')
        return False

def main(output_path: Path = OUTPUT_PATH):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    russian_stopwords = get_stopwords()

    df = load_data("SELECT title, description, category FROM news")
    if df is None or df.empty:
        logger.info(f'Data not loaded')

    df['raw_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['cleaned_text'] = clean_text_series(df['raw_text'])
    df['cleaned_text'] = remove_stopwords_from_series(df['cleaned_text'], russian_stopwords)
    df['tokens'] = tokenize_series(df['cleaned_text'])
    df['lemmas'] = lemmatize_series(df['tokens'])
    df['lemmas'] = remove_stopwords_from_tokens(df['lemmas'], russian_stopwords)
    df['lemmatized_text'] = lemmas_to_string_series(df['lemmas'])
    result_df = df[['lemmatized_text', 'category']].copy()
    success = save_data(result_df, output_path)

    if success:
        logger.info("Preprocessing completed successfully.")
    else:
        logger.error("Preprocessing failed during saving.")

if __name__ == "__main__":
    main()