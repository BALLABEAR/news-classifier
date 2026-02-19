import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

from config.parser_config import DATA_FOLDER
from scr import db

logger = logging.getLogger(__name__)

class RSSParser:
    def __init__(self, urls):
        self.urls = urls
        self.seen_links = set()

    def parse_feed(self, url):
        article_list = []
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, features='xml')
            articles = soup.find_all('item')

            for a in articles:
                title = a.find('title').text if a.find('title') else ''
                link = a.find('link').text if a.find('link') else ''
                published = a.find('pubDate').text if a.find('pubDate') else ''
                description = a.find('description').text if a.find('description') else ''

                category = ''
                if a.find('category'):
                    category = a.find('category').text
                elif a.find('rubric'):
                    category = a.find('rubric').text

                if link in self.seen_links:
                    continue
                self.seen_links.add(link)

                article = {
                    'title': title,
                    'link': link,
                    'published': published,
                    'description': description,
                    'category': category,
                    'source': url
                }
                article_list.append(article)
            logger.info(f"{url}: {len(article_list)} новостей")
        except Exception as e:
            logger.error(f"{url}: {e}")
        return article_list

    def parse_all(self):
        all_news = []
        for url in self.urls:
            news = self.parse_feed(url)
            all_news.extend(news)

            df_temp = pd.DataFrame(all_news)
            self.save(df_temp, f'{DATA_FOLDER}/raw_news_temp.csv')

        df = pd.DataFrame(all_news).drop_duplicates(subset='link')

        return df

    def save(self, df, filename=None):
        articles = df.to_dict('records')
        db.save_many_articles(articles)
        logger.info(f"Сохранено {len(articles)} новостей в БД")

        if filename:
            # filename = f'{DATA_FOLDER}/raw_news.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Дополнительно сохранено в {filename}")

        return df