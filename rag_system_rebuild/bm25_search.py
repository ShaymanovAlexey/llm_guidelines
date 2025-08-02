import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import json
import sqlite3
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
import logging

@dataclass
class SearchResult:
    """Represents a search result with document information and score."""
    doc_id: str
    title: str
    content: str
    score: float
    chunk_index: int
    metadata: Dict

class BM25Search:
    """
    BM25 (Best Matching 25) search implementation with SQLite storage.
    
    BM25 is a ranking function used by search engines to rank documents
    based on their relevance to a given search query.
    """
    
    def __init__(self, db_path: str = "bm25_database.db"):
        self.db_path = db_path
        self.k1 = 1.2  # BM25 parameter k1 (term frequency saturation)
        self.b = 0.75  # BM25 parameter b (length normalization)
        self.avgdl = 0  # Average document length
        self.doc_freq = {}  # Document frequency for each term
        self.term_freq = defaultdict(dict)  # Term frequency in each document
        self.doc_lengths = {}  # Document lengths
        self.documents = {}  # Document metadata
        self.total_docs = 0
        
        self._init_database()
        self._load_data()
    
    def _init_database(self):
        """Initialize SQLite database with necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    chunk_index INTEGER,
                    metadata TEXT,
                    length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Terms table for BM25 calculations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS terms (
                    term TEXT,
                    doc_id TEXT,
                    frequency INTEGER,
                    PRIMARY KEY (term, doc_id),
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                )
            ''')
            
            # Document frequency table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS doc_frequency (
                    term TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            ''')
            
            # Statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    key TEXT PRIMARY KEY,
                    value REAL
                )
            ''')
            
            conn.commit()
    
    def _load_data(self):
        """Load existing data from database into memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load documents
            cursor.execute('SELECT doc_id, title, content, chunk_index, metadata, length FROM documents')
            for row in cursor.fetchall():
                doc_id, title, content, chunk_index, metadata_str, length = row
                self.documents[doc_id] = {
                    'title': title,
                    'content': content,
                    'chunk_index': chunk_index,
                    'metadata': json.loads(metadata_str) if metadata_str else {},
                    'length': length
                }
                self.doc_lengths[doc_id] = length
            
            # Load term frequencies
            cursor.execute('SELECT term, doc_id, frequency FROM terms')
            for term, doc_id, freq in cursor.fetchall():
                self.term_freq[term][doc_id] = freq
            
            # Load document frequencies
            cursor.execute('SELECT term, frequency FROM doc_frequency')
            for term, freq in cursor.fetchall():
                self.doc_freq[term] = freq
            
            # Load statistics
            cursor.execute('SELECT key, value FROM statistics')
            for key, value in cursor.fetchall():
                if key == 'avgdl':
                    self.avgdl = value
                elif key == 'total_docs':
                    self.total_docs = int(value)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Convert to lowercase and split on whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Remove very short tokens (likely noise)
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def _calculate_idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term."""
        if term not in self.doc_freq or self.doc_freq[term] == 0:
            return 0
        
        return math.log((self.total_docs - self.doc_freq[term] + 0.5) / 
                       (self.doc_freq[term] + 0.5))
    
    def _calculate_bm25_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Calculate BM25 score for a document given query terms."""
        score = 0.0
        
        for term in query_terms:
            if term not in self.term_freq or doc_id not in self.term_freq[term]:
                continue
            
            # Term frequency in document
            tf = self.term_freq[term][doc_id]
            
            # Document length
            doc_len = self.doc_lengths[doc_id]
            
            # IDF calculation
            idf = self._calculate_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def add_document(self, doc_id: str, title: str, content: str, 
                    chunk_index: int = 0, metadata: Dict = None):
        """Add a document to the BM25 index."""
        if metadata is None:
            metadata = {}
        
        # Tokenize content
        terms = self._tokenize(content)
        term_counts = Counter(terms)
        
        # Update document information
        self.documents[doc_id] = {
            'title': title,
            'content': content,
            'chunk_index': chunk_index,
            'metadata': metadata,
            'length': len(terms)
        }
        self.doc_lengths[doc_id] = len(terms)
        
        # Update term frequencies
        for term, count in term_counts.items():
            self.term_freq[term][doc_id] = count
        
        # Update document frequencies
        for term in term_counts:
            if term not in self.doc_freq:
                self.doc_freq[term] = 0
            self.doc_freq[term] += 1
        
        # Update total documents
        self.total_docs += 1
        
        # Recalculate average document length
        total_length = sum(self.doc_lengths.values())
        self.avgdl = total_length / self.total_docs if self.total_docs > 0 else 0
        
        # Save to database
        self._save_document_to_db(doc_id, title, content, chunk_index, metadata, len(terms))
        self._save_terms_to_db(doc_id, term_counts)
        self._save_statistics()
    
    def _save_document_to_db(self, doc_id: str, title: str, content: str, 
                           chunk_index: int, metadata: Dict, length: int):
        """Save document to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_id, title, content, chunk_index, metadata, length)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, title, content, chunk_index, json.dumps(metadata), length))
            conn.commit()
    
    def _save_terms_to_db(self, doc_id: str, term_counts: Counter):
        """Save term frequencies to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Remove existing terms for this document
            cursor.execute('DELETE FROM terms WHERE doc_id = ?', (doc_id,))
            
            # Insert new terms
            for term, count in term_counts.items():
                cursor.execute('''
                    INSERT INTO terms (term, doc_id, frequency)
                    VALUES (?, ?, ?)
                ''', (term, doc_id, count))
            
            # Update document frequencies - only increment once per document
            # First, get existing document frequencies
            existing_freqs = {}
            cursor.execute('SELECT term, frequency FROM doc_frequency')
            for term, freq in cursor.fetchall():
                existing_freqs[term] = freq
            
            # Update frequencies for terms in this document
            for term in term_counts:
                if term in existing_freqs:
                    cursor.execute('''
                        UPDATE doc_frequency SET frequency = ? WHERE term = ?
                    ''', (existing_freqs[term] + 1, term))
                else:
                    cursor.execute('''
                        INSERT INTO doc_frequency (term, frequency) VALUES (?, ?)
                    ''', (term, 1))
            
            conn.commit()
    
    def _save_statistics(self):
        """Save statistics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM statistics')
            cursor.execute('INSERT INTO statistics (key, value) VALUES (?, ?)', ('avgdl', self.avgdl))
            cursor.execute('INSERT INTO statistics (key, value) VALUES (?, ?)', ('total_docs', self.total_docs))
            conn.commit()
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents using BM25 ranking."""
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Calculate scores for all documents
        scores = []
        for doc_id in self.documents:
            score = self._calculate_bm25_score(doc_id, query_terms)
            # Include all documents with any score (BM25 scores can be negative)
            doc_info = self.documents[doc_id]
            scores.append(SearchResult(
                doc_id=doc_id,
                title=doc_info['title'],
                content=doc_info['content'],
                score=score,
                chunk_index=doc_info['chunk_index'],
                metadata=doc_info['metadata']
            ))
        
        # Sort by score (descending) and return top_k results
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID."""
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict:
        """Get BM25 index statistics."""
        return {
            'total_documents': self.total_docs,
            'average_document_length': self.avgdl,
            'unique_terms': len(self.doc_freq),
            'total_terms': sum(self.doc_freq.values()),
            'database_size': Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        }
    
    def clear_index(self):
        """Clear all documents from the index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.term_freq.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        self.avgdl = 0
        
        # Clear database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents')
            cursor.execute('DELETE FROM terms')
            cursor.execute('DELETE FROM doc_frequency')
            cursor.execute('DELETE FROM statistics')
            conn.commit()

class AsyncBM25Search:
    """Async wrapper for BM25Search to work with async applications."""
    
    def __init__(self, db_path: str = "bm25_database.db"):
        self.bm25 = BM25Search(db_path)
    
    async def add_document(self, doc_id: str, title: str, content: str, 
                          chunk_index: int = 0, metadata: Dict = None):
        """Add a document asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.bm25.add_document, 
                                 doc_id, title, content, chunk_index, metadata)
    
    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.bm25.search, query, top_k)
    
    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.bm25.get_document, doc_id)
    
    async def get_statistics(self) -> Dict:
        """Get statistics asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.bm25.get_statistics)
    
    async def clear_index(self):
        """Clear index asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.bm25.clear_index) 