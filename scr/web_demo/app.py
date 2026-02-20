import streamlit as st
import numpy as np
import feedparser
import joblib
import json
import re
import sys
from datetime import datetime, timedelta
import dateutil.parser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import requests
from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.urls import RSS_FEEDS
from config.processing_config import CLEANING_PATTERNS

# Конфигурация страницы
st.set_page_config(page_title="Новостной классификатор", layout="wide")
st.title("📰 Поиск новостей по рубрике с нейросетевой классификацией")

# Пути к моделям и маппингу категорий
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASELINE_MODEL_DIR = BASE_DIR / "models" / "baseline_models"
NEURAL_MODEL_DIR = BASE_DIR / "models" / "neural_models"
LABEL_MAP_PATH = BASE_DIR / "data" / "processed" / "label_map.json"
TRAIN_NEURAL_PATH = BASE_DIR / "scr" / "models" / "train_neural"

# Можно использовать "svm", "lr", "lgbm", "fnn", "cnn", "rnn"
DEFAULT_MODEL_NAME = "fnn"

# Загрузка маппинга категорий
if LABEL_MAP_PATH.exists():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    categories_to_idx = {v: int(k) for k, v in label_map.items()}
    categories = list(categories_to_idx.keys())
else:
    st.error("Файл label_map.json не найден. Проверьте путь.")
    st.stop()

# Боковая панель с настройками
st.sidebar.header("Настройки")

# Выбор модели
available_models = []
# Проверяем классические модели
for pkl_file in BASELINE_MODEL_DIR.glob("model_*.pkl"):
    model_name = pkl_file.stem.replace("model_", "")
    available_models.append(f"{model_name} (classical)")
# Проверяем нейронные модели
if NEURAL_MODEL_DIR.exists():
    for model_dir in NEURAL_MODEL_DIR.iterdir():
        if model_dir.is_dir() and (model_dir / "model.pt").exists():
            available_models.append(f"{model_dir.name} (neural)")

if not available_models:
    st.error("Не найдено доступных моделей!")
    st.stop()

# Определяем модель по умолчанию
default_model_idx = 0
for i, model_name in enumerate(available_models):
    if DEFAULT_MODEL_NAME in model_name:
        default_model_idx = i
        break

selected_model_str = st.sidebar.selectbox(
    "Выберите модель",
    available_models,
    index=default_model_idx
)

# Извлекаем имя модели и тип
if " (classical)" in selected_model_str:
    selected_model_name = selected_model_str.replace(" (classical)", "")
    selected_model_type = "classical"
else:
    selected_model_name = selected_model_str.replace(" (neural)", "")
    selected_model_type = "neural"

selected_category = st.sidebar.selectbox("Выберите рубрику", categories)

period_options = {
    "За всё время": None,
    "Последний день": 1,
    "Последние 3 дня": 3,
    "Последняя неделя": 7,
    "Последний месяц": 30
}
selected_period = st.sidebar.selectbox("Период новостей", list(period_options.keys()))

# Максимальное количество новостей, которые показываем за один запуск
MAX_RESULTS = 20

def get_feed_cache() -> dict:
    if "feed_cache" not in st.session_state:
        st.session_state["feed_cache"] = {}
    return st.session_state["feed_cache"]

def get_seen_items() -> set:
    if "seen_items" not in st.session_state:
        st.session_state["seen_items"] = set()
    return st.session_state["seen_items"]

# Функции для работы с данными
@st.cache_resource
def load_model_wrapper(model_name: str, model_type: str):
    from scr.models.model_loader import load_model_by_name
    
    device = "cpu"
    return load_model_by_name(
        model_name=model_name,
        base_dir=BASE_DIR,
        device=device,
        train_neural_path=TRAIN_NEURAL_PATH
    )

@st.cache_resource
def load_lemmatizer():
    return MorphAnalyzer()

@st.cache_resource
def get_lemma_cache():
    return {}

def preprocess_text(text: str, morph: MorphAnalyzer) -> str:
    text = re.sub(CLEANING_PATTERNS['tags'], '', text)
    text = re.sub(CLEANING_PATTERNS['urls'], '', text)
    text = re.sub(CLEANING_PATTERNS['non_alpha'], '', text)
    text = text.lower()
    text = re.sub(CLEANING_PATTERNS['extra_spaces'], ' ', text).strip()

    words = text.split()

    lemma_cache = get_lemma_cache()

    lemmas = []
    for word in words:
        if word not in lemma_cache:
            lemma_cache[word] = morph.parse(word)[0].normal_form
        lemmas.append(lemma_cache[word])

    return ' '.join(lemmas)

def parse_feed_with_timeout(feed_url: str, timeout: int = 10):
    feed_cache = get_feed_cache()
    cache_entry = feed_cache.get(feed_url)

    try:
        response = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        content = response.content
        checksum = hashlib.md5(content).hexdigest()

        # Если в кеше есть тот же самый контент - используем его
        if cache_entry and cache_entry["checksum"] == checksum:
            cache_entry["timestamp"] = datetime.now()
            return cache_entry["entries"], True

        parsed = feedparser.parse(content)
        entries = []
        for entry in parsed.entries:
            # Преобразуем дату публикации, если есть
            if 'published' in entry:
                try:
                    dt = dateutil.parser.parse(entry['published'])
                    entry['datetime'] = dt.replace(tzinfo=None)
                except Exception:
                    entry['datetime'] = None
            else:
                entry['datetime'] = None
            entries.append(entry)

        # Обновляем кеш
        feed_cache[feed_url] = {
            "checksum": checksum,
            "entries": entries,
            "timestamp": datetime.now(),
        }
        return entries, False

    except requests.exceptions.Timeout:
        # Если есть старый кеш - используем его, иначе возвращаем пустой список
        if cache_entry:
            return cache_entry["entries"], True
        return [], False
    except Exception:
        if cache_entry:
            return cache_entry["entries"], True
        return [], False

def filter_entries_by_period(entries, period_name):
    days = period_options[period_name]
    if days is None:
        return entries
    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for e in entries:
        dt = e.get('datetime')
        if dt and dt >= cutoff:
            filtered.append(e)
    return filtered

def extract_clean_description(html_text: str) -> str:
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    # Если есть явные абзацы - используем их
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    if paragraphs:
        return "\n\n".join(paragraphs)

    # В противном случае — просто весь текст
    return soup.get_text(" ", strip=True)

def classify_entries_for_feed(feed_url: str, selected_period_name: str, target_idx: int,
                              model_wrapper, morph):
    entries, _ = parse_feed_with_timeout(feed_url)
    if not entries:
        return []

    period_filtered = filter_entries_by_period(entries, selected_period_name)
    if not period_filtered:
        return []

    results = []
    for entry in period_filtered:
        title = entry.get("title", "")
        raw_description = entry.get("description", "")

        display_description = extract_clean_description(raw_description)

        if not title or not display_description or not display_description.strip():
            continue

        raw_text_for_model = f"{title} {display_description}"
        processed = preprocess_text(raw_text_for_model, morph)
        
        # Используем универсальный интерфейс ModelWrapper
        pred_idx = model_wrapper.predict(processed)[0]

        if pred_idx == target_idx:
            # Оценка уверенности в виде числа [0, 1]
            proba = model_wrapper.predict_proba(processed)[0]
            confidence_score = float(proba[pred_idx])

            if confidence_score < 0.4:
                continue

            confidence_display = f"{confidence_score * 100.0:.1f}%"

            results.append({
                "id": f"{feed_url}|{entry.get('link', '')}|{title}",
                "title": title,
                "description": display_description,
                "link": entry.get("link", ""),
                "date": entry.get("published", ""),
                "datetime": entry.get("datetime"),
                "confidence": confidence_display,
                "confidence_score": confidence_score,
                "feed_url": feed_url,
            })

    return results

def render_new_cards(results, container):
    seen_items = get_seen_items()
    new_items = [r for r in results if r["id"] not in seen_items]
    if not new_items:
        return

    for r in new_items:
        seen_items.add(r["id"])
        with container:
            st.markdown(f"### [{r['title']}]({r['link']})")
            if r['description']:
                st.write(r['description'])
            col_a, col_b = st.columns(2)
            date_str = r['datetime'].strftime("%d.%m.%Y %H:%M") if r.get('datetime') else "Дата неизвестна"
            col_a.write(f"📅 {date_str}")
            col_b.write(f"🎯 Уверенность: {r['confidence']}")
            st.divider()

# Основной блок: обработка нажатия кнопки
if st.sidebar.button("Показать новости"):
    # Сбрасываем состояние для нового запуска
    st.session_state["seen_items"] = set()
    st.session_state["filtered_results"] = []

    # Загружаем модель через универсальный загрузчик
    model_wrapper = load_model_wrapper(selected_model_name, selected_model_type)
    morph = load_lemmatizer()
    target_idx = categories_to_idx[selected_category]

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Контейнер для "потокового" вывода карточек
    stream_container = st.container()

    all_results = []

    # Параллельно обрабатываем фиды, но UI обновляем по мере завершения каждого
    with ThreadPoolExecutor(max_workers=min(4, len(RSS_FEEDS))) as executor:
        future_to_feed = {
            executor.submit(
                classify_entries_for_feed,
                feed_url,
                selected_period,
                target_idx,
                model_wrapper,
                morph,
            ): feed_url
            for feed_url in RSS_FEEDS
        }

        total = len(future_to_feed)
        completed = 0

        for future in as_completed(future_to_feed):
            feed_url = future_to_feed[future]
            completed += 1

            try:
                feed_results = future.result()
            except Exception as e:
                st.sidebar.warning(f"⚠️ Ошибка обработки {feed_url}: {e}")
                feed_results = []

            status_text.text(f"Обработано {completed}/{total} лент")
            progress_bar.progress(completed / total)

            if len(all_results) >= MAX_RESULTS:
                continue

            status_text.text(f"Обработано {completed}/{total} лент")
            progress_bar.progress(completed / total)

            if feed_results and len(all_results) < MAX_RESULTS:
                # Добавляем в общий список и сразу показываем только новые карточки,
                # пока не достигнут лимит MAX_RESULTS
                for item in feed_results:
                    if len(all_results) >= MAX_RESULTS:
                        break
                    all_results.append(item)
                    render_new_cards([item], stream_container)

    progress_bar.empty()
    status_text.empty()

    st.session_state.filtered_results = all_results[:MAX_RESULTS]

# Отображение результатов
if "filtered_results" in st.session_state and st.session_state.filtered_results:
    results = st.session_state.filtered_results
    st.success(f"Найдено {len(results)} новостей в рубрике «{selected_category}»")

# Подсказка, если ничего не найдено
elif "filtered_results" in st.session_state and not st.session_state.filtered_results:
    st.warning("Нет новостей, соответствующих выбранной рубрике.")