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
import datetime
BRAVE_PATH = "/usr/bin/brave-browser"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"
USER_DATA_DIR = "/home/alex/.config/BraveSoftware/Brave-Browser"
PROFILE_DIR = "Default"
BASE_URL = "https://www.ainvest.com"
NEWS_URL = "https://www.ainvest.com/news/articles-latest/"
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))
from fuzzy_vector_store import FuzzyVectorStore
from summary_generator import generate_summary
from config import SUMMARY_CONFIG, VECTOR_STORE_CONFIG, NEWS_SOURCES, get_summary_kwargs, BM25_CONFIG
from bm25_database import BM25Database
import chromadb
from chromadb.config import Settings

def ensure_collection_exists():
    """Ensure the collection exists and is visible in ChromaDB."""
    try:
        client = chromadb.PersistentClient(
            path=VECTOR_STORE_CONFIG['persist_directory'],
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Try to get the collection - if it doesn't exist, create it
        try:
            collection = client.get_collection(VECTOR_STORE_CONFIG['collection_name'])
            print(f"✅ Collection '{VECTOR_STORE_CONFIG['collection_name']}' already exists")
        except:
            # Collection doesn't exist, create it
            collection = client.create_collection(VECTOR_STORE_CONFIG['collection_name'])
            print(f"✅ Created collection '{VECTOR_STORE_CONFIG['collection_name']}'")
        
        return True
    except Exception as e:
        print(f"❌ Error ensuring collection exists: {e}")
        return False

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
        # Ensure collection exists and is visible
        if not ensure_collection_exists():
            print("Failed to ensure collection exists. Exiting.")
            return
        
        # Initialize both storage systems
        print("Initializing storage systems...")
        store = FuzzyVectorStore(
            collection_name=VECTOR_STORE_CONFIG['collection_name'],
            persist_directory=VECTOR_STORE_CONFIG['persist_directory'],
            duplicate_threshold=0.9  # Fuzzy duplicate detection threshold
        )
        
        bm25_db = BM25Database(
            database_path=BM25_CONFIG['database_path'],
            collection_name=BM25_CONFIG['collection_name']
        )
        
        # Get configuration
        summary_generator_type = SUMMARY_CONFIG['generator_type']
        max_length = SUMMARY_CONFIG['max_length']
        summary_kwargs = get_summary_kwargs()
        
        news = extract_latest_news_selenium()
        
        print(f"Using summary generator: {summary_generator_type}")
        print(f"Summary length: {max_length} characters")
        print(f"Vector store path: {VECTOR_STORE_CONFIG['persist_directory']}")
        print(f"BM25 database path: {BM25_CONFIG['database_path']}")
        print(f"Fuzzy duplicate threshold: {store.duplicate_threshold}")
        
        # Track statistics
        vector_docs_added = 0
        bm25_docs_added = 0
        
        for i, article in enumerate(news, 1):
            if not article['content']:
                continue
                
            # Generate metadata with timestamp and summary
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate summary using the configured generator
            summary = await generate_summary(
                article['content'], 
                max_length=max_length, 
                generator_type=summary_generator_type, 
                **summary_kwargs
            )
            
            # Prepare document for both storage systems
            doc = {
                'text': article['content'],
                'metadata': {
                    'title': article['title'],
                    'url': article['url'],
                    'source': 'ainvest.com',
                    'timestamp': timestamp,
                    'created_at': timestamp,
                    'summary': summary,
                    'topic': NEWS_SOURCES['ainvest']['topic'],
                    'content_length': len(article['content']),
                    'extraction_method': 'selenium',
                    'summary_generator': summary_generator_type
                }
            }
            
            # Add to vector store (with fuzzy duplicate detection)
            try:
                await store.add_documents([doc])
                vector_docs_added += 1
                print(f"✅ Added to vector store (fuzzy duplicate detection enabled)")
            except Exception as e:
                print(f"❌ Error adding to vector store: {e}")
            
            # Add to BM25 database
            try:
                if bm25_db.add_document(doc):
                    bm25_docs_added += 1
                    print(f"✅ Added to BM25 database")
                else:
                    print(f"❌ Failed to add to BM25 database")
            except Exception as e:
                print(f"❌ Error adding to BM25 database: {e}")
            
            print(f"\n--- Article {i} ---")
            print(f"Title: {article['title']}")
            print(f"URL: {article['url']}")
            print(f"Summary: {summary}")
            print(f"Content length: {len(article['content'])} characters")
            print(f"Summary generator: {summary_generator_type}")
            print("-" * 50)
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)
        
        # Vector store stats
        vector_stats = await store.get_collection_stats()
        print(f"🔍 Fuzzy Vector Store:")
        print(f"   • Total documents: {vector_stats['total_documents']}")
        print(f"   • Documents added this run: {vector_docs_added}")
        print(f"   • Duplicate threshold: {store.duplicate_threshold}")
        
        # BM25 database stats
        bm25_stats = bm25_db.get_stats()
        print(f"📚 BM25 Database:")
        print(f"   • Total documents: {bm25_stats.get('total_documents', 0)}")
        print(f"   • Documents added this run: {bm25_docs_added}")
        print(f"   • Documents by source: {bm25_stats.get('documents_by_source', {})}")
        
        print(f"\n✅ SUCCESS: Documents saved to both storage systems!")
        print(f"   • Fuzzy vector embeddings: {VECTOR_STORE_CONFIG['persist_directory']}")
        print(f"   • BM25 database: {BM25_CONFIG['database_path']}")
        
    asyncio.run(main()) 