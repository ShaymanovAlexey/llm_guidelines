import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Set

BITCOIN_COM_URL = 'https://news.bitcoin.com/'
AINVEST_URL = 'https://www.ainvest.com/news/'


def fetch_bitcoin_com_news() -> List[Dict[str, Any]]:
    """Fetch and parse news from news.bitcoin.com homepage."""
    resp = requests.get(BITCOIN_COM_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    articles = []
    # Find main news articles (example: headlines)
    for item in soup.select('article')[:5]:  # Limit to 5 for demo
        title = item.find('h3') or item.find('h2')
        title_text = title.get_text(strip=True) if title else ''
        link = item.find('a')
        url = link['href'] if link and link.has_attr('href') else BITCOIN_COM_URL
        summary = item.find('p')
        summary_text = summary.get_text(strip=True) if summary else ''
        articles.append({
            'text': f"{title_text}\n{summary_text}",
            'metadata': {
                'source': 'news.bitcoin.com',
                'url': url
            }
        })
    return articles

def fetch_ainvest_news() -> List[Dict[str, Any]]:
    """Fetch and parse the specific AInvest news article."""
    resp = requests.get(AINVEST_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Extract title
    title = soup.find('h1') or soup.find('h2')
    title_text = title.get_text(strip=True) if title else ''
    # Extract main content (try to find the main article body)
    content = ''
    article_body = soup.find('article') or soup.find('div', class_='article')
    if article_body:
        paragraphs = article_body.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    else:
        # Fallback: get all <p> tags
        paragraphs = soup.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    articles = [{
        'text': f"{title_text}\n{content}",
        'metadata': {
            'source': 'ainvest.com',
            'url': AINVEST_URL
        }
    }]
    return articles

def fetch_and_parse_article(url: str, topic: str = None) -> Dict[str, Any]:
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Extract title
    title = soup.find('h1') or soup.find('h2')
    title_text = title.get_text(strip=True) if title else ''
    # Extract main content (try to find the main article body)
    content = ''
    article_body = soup.find('article') or soup.find('div', class_='article')
    if article_body:
        paragraphs = article_body.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    else:
        # Fallback: get all <p> tags
        paragraphs = soup.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    # Only return if topic is in title or content (case-insensitive)
    if topic:
        topic_lower = topic.lower()
        if topic_lower not in title_text.lower() and topic_lower not in content.lower():
            return None
    # Skip empty articles
    if not title_text.strip() and not content.strip():
        return None
    return {
        'text': f"{title_text}\n{content}",
        'metadata': {
            'source': urlparse(url).netloc,
            'url': url
        }
    }

def extract_article_context(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        article_body = soup.find('article') or soup.find('div', class_='article')
        if article_body:
            paragraphs = article_body.find_all('p')
            return '\n'.join(p.get_text(strip=True) for p in paragraphs)
        else:
            paragraphs = soup.find_all('p')
            return '\n'.join(p.get_text(strip=True) for p in paragraphs)
    except Exception as e:
        return f"[Error fetching context: {e}]"

def extract_news_from_html(html: str, base_url: str = '') -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    for tag in soup.find_all(attrs={'data-title': True}):
        title = tag['data-title']
        url = tag['href'] if tag.name == 'a' and tag.has_attr('href') else None
        if url and base_url:
            url = urljoin(base_url, url)
        date_tag = tag.find_next(attrs={'itemprop': 'datePublished'})
        date = date_tag['content'] if date_tag and date_tag.has_attr('content') else None
        author_tag = tag.find_next(attrs={'itemprop': 'name'})
        author = author_tag.get_text(strip=True) if author_tag else None
        context = extract_article_context(url) if url else ''
        articles.append({
            'title': title,
            'url': url,
            'date': date,
            'author': author,
            'context': context
        })
    return articles

def crawl_site(base_url: str, topic: str = None, max_links: int = 10) -> List[Dict[str, Any]]:
    """Crawl a site and its internal links (depth=1), extract news with data-title and fetch context."""
    try:
        resp = requests.get(base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        domain = urlparse(base_url).netloc
        links: Set[str] = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == domain and full_url.startswith('http'):
                links.add(full_url)
        links = list(links)[:max_links]
        all_urls = [base_url] + links
        articles = []
        for url in all_urls:
            try:
                page_resp = requests.get(url, timeout=10)
                page_resp.raise_for_status()
                page_html = page_resp.text
                page_articles = extract_news_from_html(page_html, base_url=url)
                # Optionally filter by topic
                if topic:
                    topic_lower = topic.lower()
                    page_articles = [art for art in page_articles if topic_lower in art['title'].lower() or topic_lower in art['context'].lower()]
                articles.extend(page_articles)
            except Exception as e:
                articles.append({'title': '', 'url': url, 'date': None, 'author': None, 'context': f'[Error: {e}]'})
        return articles
    except Exception as e:
        return [{'title': '', 'url': base_url, 'date': None, 'author': None, 'context': f'[Error: {e}]'}]

def fetch_news_from_sources(urls: List[str], topic: str = None) -> List[Dict[str, Any]]:
    """Fetch and combine news from a list of sites and their subsites about a topic."""
    news = []
    for url in urls:
        news.extend(crawl_site(url, topic=topic))
    return news

def debug_crawl_site(base_url: str, topic: str = None, max_links: int = 10):
    """Debug crawl: print all crawled URLs, titles, and content snippets, and topic match status."""
    try:
        resp = requests.get(base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        domain = urlparse(base_url).netloc
        links: Set[str] = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == domain and full_url.startswith('http'):
                links.add(full_url)
        links = list(links)[:max_links]
        all_urls = [base_url] + links
        for url in all_urls:
            try:
                resp2 = requests.get(url)
                resp2.raise_for_status()
                soup2 = BeautifulSoup(resp2.text, 'html.parser')
                title = soup2.find('h1') or soup2.find('h2')
                title_text = title.get_text(strip=True) if title else ''
                article_body = soup2.find('article') or soup2.find('div', class_='article')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                else:
                    paragraphs = soup2.find_all('p')
                    content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                topic_match = False
                if topic:
                    topic_lower = topic.lower()
                    if topic_lower in title_text.lower() or topic_lower in content.lower():
                        topic_match = True
                print(f"\nURL: {url}")
                print(f"Title: {title_text}")
                print(f"Content snippet: {content[:120]}...")
                print(f"Topic match: {topic_match}")
            except Exception as e:
                print(f"\nURL: {url}")
                print(f"Error: {e}")
    except Exception as e:
        print(f"Error crawling {base_url}: {e}") 