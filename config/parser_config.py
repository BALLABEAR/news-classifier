import os
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = 'data'

DB_CONFIG = {
    "dbname": "news_db",
    "user": "news_user",
    "password": os.getenv("DB_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}