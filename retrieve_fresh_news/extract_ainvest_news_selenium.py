from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import time
import asyncio
BRAVE_PATH = "/usr/bin/brave-browser"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"
USER_DATA_DIR = "/home/alex/.config/BraveSoftware/Brave-Browser"
PROFILE_DIR = "Default"
BASE_URL = "https://www.ainvest.com"
NEWS_URL = "https://www.ainvest.com/news/articles-latest/"
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))
from vector_store import VectorStore

def extract_article_content(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Try to extract main content heuristically
        article = soup.find('article')
        if article:
            return article.get_text(separator=' ', strip=True)
        # Fallback: get largest <div> by text length
        divs = soup.find_all('div')
        if divs:
            largest = max(divs, key=lambda d: len(d.get_text()))
            return largest.get_text(separator=' ', strip=True)
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"[Error fetching content: {e}]"

def extract_latest_news_selenium():
    options = Options()
    options.binary_location = BRAVE_PATH
    # options.add_argument(f'--user-data-dir={USER_DATA_DIR}')
    # options.add_argument(f'--profile-directory={PROFILE_DIR}')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(NEWS_URL)
    try:
        # Wait for at least one element with data-title to appear
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-title]'))
        )
    except Exception:
        time.sleep(10)  # fallback wait
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    for tag in soup.find_all(attrs={'data-title': True}):
        title = tag['data-title']
        url = tag['href'] if tag.has_attr('href') else None
        if url:
            url = urljoin(BASE_URL, url)
            content = extract_article_content(url)
        else:
            content = None
        articles.append({'title': title, 'url': url, 'content': content})
    return articles

if __name__ == "__main__":
    async def main():
        store = VectorStore(persist_directory = "./rag_system_rebuild/chroma_db")
        news = extract_latest_news_selenium()
        for i, article in enumerate(news, 1):
            docs = [
                {'text': article['content']}
            ]
            await store.add_documents(docs, chunk_size=1000, chunk_overlap=200, topic=article['title'])
            all_docs = await store.list_documents()
            # print(f"\n--- Article {i} ---")
            # print(f"Title: {article['title']}")
            # print(f"URL: {article['url']}")
            # print(f"Content: {article['content'][:800]}...")  # Show first 800 chars
            # print("-" * 50)
    asyncio.run(main()) 