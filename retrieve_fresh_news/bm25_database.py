#!/usr/bin/env python3
"""
BM25 Database Handler for storing documents alongside vector embeddings.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any
from datetime import datetime

class BM25Database:
    """Handles BM25 database operations for document storage."""
    
    def __init__(self, database_path: str, collection_name: str = "news_documents"):
        self.database_path = database_path
        self.collection_name = collection_name
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure the database and tables exist."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT,
                source TEXT,
                timestamp TEXT,
                summary TEXT,
                topic TEXT,
                content_length INTEGER,
                extraction_method TEXT,
                summary_generator TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster searches
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_documents_source 
            ON documents(source)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_documents_timestamp 
            ON documents(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """Add a document to the BM25 database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Extract metadata
            metadata = document.get('metadata', {})
            
            cursor.execute('''
                INSERT INTO documents (
                    title, content, url, source, timestamp, summary, topic,
                    content_length, extraction_method, summary_generator, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.get('title', ''),
                document.get('text', ''),
                metadata.get('url', ''),
                metadata.get('source', ''),
                metadata.get('timestamp', ''),
                metadata.get('summary', ''),
                metadata.get('topic', ''),
                metadata.get('content_length', 0),
                metadata.get('extraction_method', ''),
                metadata.get('summary_generator', ''),
                json.dumps(metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error adding document to BM25 database: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add multiple documents to the BM25 database."""
        success_count = 0
        for doc in documents:
            if self.add_document(doc):
                success_count += 1
        return success_count
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"❌ Error getting document count: {e}")
            return 0
    
    def get_documents(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Get documents from the database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT title, content, url, source, timestamp, summary, topic, metadata
                FROM documents 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'title': row[0],
                    'content': row[1],
                    'url': row[2],
                    'source': row[3],
                    'timestamp': row[4],
                    'summary': row[5],
                    'topic': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {}
                })
            
            conn.close()
            return documents
            
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using simple text matching."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Simple text search in title, content, and summary
            cursor.execute('''
                SELECT title, content, url, source, timestamp, summary, topic, metadata
                FROM documents 
                WHERE title LIKE ? OR content LIKE ? OR summary LIKE ?
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'title': row[0],
                    'content': row[1],
                    'url': row[2],
                    'source': row[3],
                    'timestamp': row[4],
                    'summary': row[5],
                    'topic': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {}
                })
            
            conn.close()
            return documents
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Total documents
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_documents = cursor.fetchone()[0]
            
            # Documents by source
            cursor.execute('SELECT source, COUNT(*) FROM documents GROUP BY source')
            documents_by_source = dict(cursor.fetchall())
            
            # Recent documents
            cursor.execute('SELECT COUNT(*) FROM documents WHERE created_at >= datetime("now", "-1 day")')
            recent_documents = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_documents': total_documents,
                'documents_by_source': documents_by_source,
                'recent_documents': recent_documents,
                'database_path': self.database_path
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {} 