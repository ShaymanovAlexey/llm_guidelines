from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))
from vector_store import VectorStore

BRAVE_PATH = "/usr/bin/brave-browser"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"
BASE_URL = "https://news.bitcoin.com"
NEWS_URL = "https://news.bitcoin.com/latest-news"

def extract_article_content_selenium(driver, url):
    try:
        print(f"Fetching content from: {url}")
        driver.get(url)
        time.sleep(8)  # Wait longer for JS to load
        
        # Try multiple selectors for content, prioritizing article content
        content_selectors = [
            '.post-content p',  # Paragraphs in post content
            '.entry-content p',  # Paragraphs in entry content
            'article p',         # Paragraphs in article
            '.content p',        # Paragraphs in content
            '.post-body p',      # Paragraphs in post body
            '.article-body p',   # Paragraphs in article body
            'main p',            # Paragraphs in main
            '.main-content p',   # Paragraphs in main content
            '.post-content',     # Post content container
            '.entry-content',    # Entry content container
            'article',           # Article container
            '.content',          # Content container
            'main',              # Main container
            '.main-content'      # Main content container
        ]
        
        for selector in content_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    # Collect text from all matching elements
                    texts = []
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) > 20:  # Only meaningful paragraphs
                            texts.append(text)
                    
                    if texts:
                        # Join all paragraphs
                        full_text = ' '.join(texts)
                        if len(full_text) > 200:  # Ensure we got substantial content
                            print(f"Found content using selector: {selector}")
                            return full_text
            except Exception as e:
                continue
        
        # Fallback: try to get the largest text block
        try:
            # Look for divs with substantial text content
            divs = driver.find_elements(By.TAG_NAME, 'div')
            largest_text = ""
            for div in divs:
                text = div.text.strip()
                if len(text) > len(largest_text) and len(text) > 500:
                    # Check if it looks like article content (not ads/nav)
                    if not any(ad_word in text.lower() for ad_word in ['bonus', 'casino', 'gaming', 'bet', 'review', 'get bonus']):
                        largest_text = text
            
            if largest_text:
                return largest_text
        except:
            pass
        
        # Last resort: get body text but filter out ads
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
            body_text = body.text
            # Split into lines and filter out ad-like content
            lines = body_text.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:
                    # Skip lines that look like ads
                    if not any(ad_word in line.lower() for ad_word in ['bonus', 'casino', 'gaming', 'bet', 'review', 'get bonus', 'up to']):
                        filtered_lines.append(line)
            
            if filtered_lines:
                return ' '.join(filtered_lines)
        except:
            pass
        
        return "[Could not extract meaningful content]"
    except Exception as e:
        return f"[Error fetching content: {e}]"

def extract_title_from_link(a_tag):
    """Extract title from various sources within a link tag"""
    # First try: heading tags
    h = a_tag.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if h:
        title = h.get_text(strip=True)
        if title and len(title) > 10:
            return title
    
    # Second try: img alt attribute
    img = a_tag.find('img')
    if img and img.get('alt'):
        alt_text = img['alt'].strip()
        if alt_text and len(alt_text) > 10:
            return alt_text
    
    # Third try: direct link text
    link_text = a_tag.get_text(strip=True)
    if link_text and len(link_text) > 10:
        return link_text
    
    # Fourth try: any text content within the link
    for child in a_tag.children:
        if hasattr(child, 'get_text'):
            text = child.get_text(strip=True)
            if text and len(text) > 10:
                return text
    
    return None

def extract_latest_bitcoin_news_selenium():
    options = Options()
    options.binary_location = BRAVE_PATH
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(NEWS_URL)
    
    # Wait longer for dynamic content
    print("Waiting for page to load...")
    time.sleep(15)
    
    try:
        # Wait for various possible selectors
        selectors_to_try = [
            'a[href*="/20"]',  # Links with year in URL (likely articles)
            'a[href*="/"]',    # Any links with slashes
            'article a',       # Links inside articles
            '.post a',         # Links inside posts
            'a'                # Any links
        ]
        
        for selector in selectors_to_try:
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                print(f"Found elements with selector: {selector}")
                break
            except:
                continue
    except Exception as e:
        print(f"Wait error: {e}")
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    processed_urls = set()  # Track processed URLs to avoid duplicates
    
    # Remove footer section to avoid footer links
    footer = soup.find('footer')
    if footer:
        footer.decompose()
    
    # Also remove any elements with footer-related classes
    for elem in soup.find_all(class_=lambda x: x and 'footer' in x.lower()):
        elem.decompose()
    
    # Find all links that might contain news (excluding footer)
    all_links = soup.find_all('a', href=True)
    print(f"Found {len(all_links)} total links (footer excluded)")
    
    # Debug: show more links and search for specific article
    print("\nExample links found:")
    for i, a in enumerate(all_links[:30]):
        href = a['href']
        title = extract_title_from_link(a)
        print(f"  {i+1}. {title} -> {href}")
        
        # Check for the El Salvador article
        if 'latam-insights' in href or 'el-salvador' in href:
            print(f"  *** FOUND EL SALVADOR ARTICLE: {title} -> {href} ***")
    
    # Also search for any links containing specific keywords
    print("\nSearching for specific articles:")
    for a in all_links:
        href = a['href']
        title = extract_title_from_link(a)
        if any(keyword in href.lower() for keyword in ['latam', 'salvador', 'brazil', 'imf']):
            print(f"  Found: {title} -> {href}")
    
    for a in all_links:
        href = a['href']
        
        # Skip non-article links
        if any(skip in href.lower() for skip in ['#', 'javascript:', 'mailto:', 'tel:']):
            continue
        
        # Handle relative URLs
        if not href.startswith('http'):
            href = urljoin(BASE_URL, href)
        elif not href.startswith(BASE_URL):
            # Skip external links
            continue
        
        # Skip category pages and other non-article pages
        if any(skip in href.lower() for skip in ['/category/', '/submit-', '/press-', '/tag/', '/author/', '/page/']):
            continue
        
        # Skip homepage
        if href == BASE_URL or href == BASE_URL + '/':
            continue
            
        # Skip if already processed
        if href in processed_urls:
            continue
            
        # Get title using improved extraction
        title = extract_title_from_link(a)
        
        if title and len(title) > 10:  # Ensure we have a meaningful title
            print(f"Processing: {title} -> {href}")
            content = extract_article_content_selenium(driver, href)
            articles.append({'title': title, 'url': href, 'content': content})
            processed_urls.add(href)  # Mark as processed
    
    driver.quit()
    print(f"Extracted {len(articles)} unique articles")
    return articles

if __name__ == "__main__":
    async def main():
        store = VectorStore(persist_directory = "./rag_system_rebuild/chroma_db")
        news = extract_latest_bitcoin_news_selenium()
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