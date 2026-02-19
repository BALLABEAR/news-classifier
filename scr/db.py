import psycopg2
from config.parser_config import DB_CONFIG
import logging
logger = logging.getLogger(__name__)

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id SERIAL PRIMARY KEY,
            link TEXT UNIQUE,
            published TEXT,
            description TEXT,
            category TEXT,
            source TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_article(article):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO news (title, link, published, description, category, source)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (link) DO NOTHING;
    """, (
        article['title'],
        article['link'],
        article['published'],
        article['description'],
        article['category'],
        article['source']
    ))
    conn.commit()
    cur.close()
    conn.close()

def save_many_articles(articles):
    conn = get_connection()
    cur = conn.cursor()
    for article in articles:
        try:
            cur.execute("""
                INSERT INTO news (title, link, published, description, category, source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (link) DO NOTHING;
            """, (
                article['title'],
                article['link'],
                article['published'],
                article['description'],
                article['category'],
                article['source']
            ))
        except Exception as e:
            logger.error(f'Ошибка вставки {article['link']}: {e}')
    conn.commit()
    cur.close()
    conn.close()
