# Project for LLM

A comprehensive project containing RAG (Retrieval-Augmented Generation) systems and news retrieval tools.

## Project Structure

### `rag_system_rebuild/`
Advanced RAG system implementation with:
- **ChromaDB Integration**: Vector database for document storage and retrieval
- **Document Processing**: Text extraction and processing capabilities
- **Ollama Integration**: Local LLM generation using Ollama models
- **Fuzzy Vector Store**: Enhanced search with fuzzy matching
- **Web Interface**: HTML templates for user interaction

Key files:
- `main.py` - Main application entry point
- `advanced_rag_system.py` - Advanced RAG implementation
- `simple_rag_system.py` - Basic RAG system
- `vector_store.py` - Vector storage management
- `ollama_generator.py` - LLM generation interface

### `retrieve_fresh_news/`
News retrieval and processing system:
- **Selenium-based Scraping**: Automated news extraction from various sources
- **AI Investment News**: Specialized extraction for AI investment news
- **Bitcoin News**: Cryptocurrency news retrieval
- **HTML Processing**: Test utilities for HTML content processing

Key files:
- `news_retriever.py` - Main news retrieval interface
- `extract_ainvest_news_selenium.py` - AI investment news extraction
- `extract_bitcoin_news_selenium.py` - Bitcoin news extraction
- `__main__.py` - Command-line interface

## Features

### RAG System
- Document ingestion and processing
- Vector-based similarity search
- Local LLM integration with Ollama
- Web-based user interface
- Fuzzy search capabilities
- Performance metrics tracking

### News Retrieval
- Automated web scraping with Selenium
- Multiple news source support
- HTML content extraction and processing
- Test utilities for validation
- Modular architecture for easy extension

## Setup

### Prerequisites
- Python 3.8+
- Ollama (for local LLM generation)
- Brave browser (for Selenium scraping)

### Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv rag_system
   source rag_system/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r rag_system_rebuild/requirements.txt
   pip install -r retrieve_fresh_news/requirements.txt
   ```

### Usage

#### RAG System
```bash
cd rag_system_rebuild
python main.py
```

#### News Retrieval
```bash
cd retrieve_fresh_news
python -m retrieve_fresh_news
```

## Development

The project uses a Python virtual environment named `rag_system` for all development work. Make sure to activate it before running any commands.

## Testing

Both modules include comprehensive test suites:
- `test_*.py` files in `rag_system_rebuild/`
- `test_*.py` files in `retrieve_fresh_news/`

Run tests with:
```bash
python -m pytest
```

## License

This project is for educational and research purposes. 