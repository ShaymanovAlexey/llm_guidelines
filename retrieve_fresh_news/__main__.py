from .news_retriever import fetch_news_from_sources, debug_crawl_site
import sys

DEFAULT_URLS = [
    'https://news.bitcoin.com/',
    'https://www.ainvest.com/news/'
]

def main():
    args = sys.argv[1:]
    urls = []
    topic = None
    debug = False
    # Check for --debug flag
    if '--debug' in args:
        debug = True
        args.remove('--debug')
    # If the last argument does not look like a URL, treat it as the topic
    if args and not args[0].startswith('http'):
        topic = args[0]
        urls = DEFAULT_URLS
    elif args:
        urls = [a for a in args if a.startswith('http')]
        if len(args) > len(urls):
            topic = args[-1]
    else:
        urls = DEFAULT_URLS
    print(f"Fetching news from: {urls}")
    if topic:
        print(f"Filtering for topic: {topic}")
    if debug:
        for url in urls:
            debug_crawl_site(url, topic=topic)
        return
    try:
        news = fetch_news_from_sources(urls, topic=topic)
        for i, article in enumerate(news, 1):
            print(f"\n--- Article {i} ---")
            print(article['text'])
            print(f"Metadata: {article['metadata']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 