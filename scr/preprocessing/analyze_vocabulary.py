import pandas as pd
from collections import Counter
from pathlib import Path

# Путь к файлу
BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "news_clean.csv"


def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Загружено {len(df)} строк")
    return df


def get_all_words(df):
    # Объединяем все тексты
    all_text = ' '.join(df['lemmatized_text'].astype(str))
    # Разбиваем на слова
    words = all_text.split()
    return words


def analyze_words(words, top_n=100):
    # Подсчет частоты
    word_counts = Counter(words)

    print(f"\n{'=' * 50}")
    print(f"Всего уникальных слов: {len(word_counts)}")
    print(f"Всего слов (с повторениями): {len(words)}")
    print(f"{'=' * 50}\n")

    # Топ самых частых
    print(f"Топ-{top_n} самых частых слов:")
    for word, count in word_counts.most_common(top_n):
        print(f"{word}: {count}")

    # Самые редкие (по одному разу)
    rare_words = [word for word, count in word_counts.items() if count == 1]
    print(f"\nСлов, встречающихся 1 раз: {len(rare_words)}")
    print("Примеры редких слов:", rare_words[:20])

    return word_counts


def find_potential_stopwords(word_counts, min_freq=1000):
    """Находит потенциальные стоп-слова (очень частые)"""
    potential_stopwords = []
    for word, count in word_counts.most_common():
        if count > min_freq and len(word) > 2:
            potential_stopwords.append(word)

    print(f"\nПотенциальные стоп-слова (> {min_freq} вхождений):")
    print(potential_stopwords[:30])
    return potential_stopwords


def find_website_names(word_counts):
    """Ищет названия сайтов и характерные для новостей слова"""
    website_indicators = ['.ru', '.com', 'http', 'www', 'ria', 'lenta', 'tass', 'interfax']
    website_words = []

    for word in word_counts:
        if any(indicator in word.lower() for indicator in website_indicators):
            website_words.append((word, word_counts[word]))

    website_words.sort(key=lambda x: x[1], reverse=True)

    print(f"\nСлова, похожие на названия сайтов:")
    for word, count in website_words[:20]:
        print(f"{word}: {count}")

    return website_words


def save_results(word_counts, output_path="vocabulary_analysis.txt"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Всего уникальных слов: {len(word_counts)}\n")
        f.write(f"Всего слов (с повторениями): {sum(word_counts.values())}\n\n")

        f.write("Топ-500 самых частых слов:\n")
        for word, count in word_counts.most_common(500):
            f.write(f"{word}: {count}\n")

    print(f"\nРезультаты сохранены в {output_path}")


def main():
    df = load_data()
    words = get_all_words(df)
    word_counts = analyze_words(words, top_n=100)

    find_potential_stopwords(word_counts)
    find_website_names(word_counts)

    save_results(word_counts)

    # Поиск конкретных слов
    search_words = ['фото', 'видео', 'смотреть', 'ria', 'lenta', 'путин', 'украина']
    print(f"\nЧастота конкретных слов:")
    for word in search_words:
        print(f"{word}: {word_counts.get(word, 0)}")


if __name__ == "__main__":
    main()