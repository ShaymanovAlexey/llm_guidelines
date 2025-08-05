"""
Airflow DAG for automated news extraction from multiple sources.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
import os
import sys

# Add the project path to Python path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_path)

# Default arguments for the DAG
default_args = {
    'owner': 'news_extraction_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Create the DAG
dag = DAG(
    'news_extraction_pipeline',
    default_args=default_args,
    description='Automated news extraction from AI Investment and Bitcoin news sources',
    schedule_interval='0 */6 * * *',  # Run every 6 hours
    max_active_runs=1,
    tags=['news', 'extraction', 'ai', 'bitcoin'],
)

def extract_ainvest_news():
    """Extract news from AI Investment website."""
    import subprocess
    import sys
    
    # Path to the news extraction script
    script_path = os.path.join(project_path, 'retrieve_fresh_news', 'extract_ainvest_news_selenium.py')
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=project_path)
        
        if result.returncode == 0:
            print("✅ AI Investment news extraction completed successfully")
            print(result.stdout)
            return "SUCCESS"
        else:
            print(f"❌ AI Investment news extraction failed: {result.stderr}")
            raise Exception(f"Script failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running AI Investment news extraction: {e}")
        raise

def extract_bitcoin_news():
    """Extract news from Bitcoin news website."""
    import subprocess
    import sys
    
    # Path to the news extraction script
    script_path = os.path.join(project_path, 'retrieve_fresh_news', 'extract_bitcoin_news_selenium.py')
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=project_path)
        
        if result.returncode == 0:
            print("✅ Bitcoin news extraction completed successfully")
            print(result.stdout)
            return "SUCCESS"
        else:
            print(f"❌ Bitcoin news extraction failed: {result.stderr}")
            raise Exception(f"Script failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running Bitcoin news extraction: {e}")
        raise

def check_database_stats():
    """Check the statistics of both vector store and BM25 database."""
    import asyncio
    import sys
    
    # Add the rag_system_rebuild path
    rag_path = os.path.join(project_path, 'rag_system_rebuild')
    sys.path.insert(0, rag_path)
    
    # Add the retrieve_fresh_news path
    news_path = os.path.join(project_path, 'retrieve_fresh_news')
    sys.path.insert(0, news_path)
    
    async def get_stats():
        try:
            from fuzzy_vector_store import FuzzyVectorStore
            from bm25_database import BM25Database
            from config import VECTOR_STORE_CONFIG, BM25_CONFIG
            
            # Get vector store stats
            store = FuzzyVectorStore(
                collection_name=VECTOR_STORE_CONFIG['collection_name'],
                persist_directory=VECTOR_STORE_CONFIG['persist_directory']
            )
            vector_stats = await store.get_collection_stats()
            
            # Get BM25 database stats
            bm25_db = BM25Database(
                database_path=BM25_CONFIG['database_path'],
                collection_name=BM25_CONFIG['collection_name']
            )
            bm25_stats = bm25_db.get_stats()
            
            print("📊 DATABASE STATISTICS")
            print("=" * 50)
            print(f"🔍 Fuzzy Vector Store:")
            print(f"   • Total documents: {vector_stats['total_documents']}")
            print(f"   • Collection: {vector_stats['collection_name']}")
            
            print(f"\n📚 BM25 Database:")
            print(f"   • Total documents: {bm25_stats.get('total_documents', 0)}")
            print(f"   • Documents by source: {bm25_stats.get('documents_by_source', {})}")
            print(f"   • Recent documents: {bm25_stats.get('recent_documents', 0)}")
            
            return "SUCCESS"
            
        except Exception as e:
            print(f"❌ Error checking database stats: {e}")
            raise
    
    return asyncio.run(get_stats())

# Define tasks
extract_ainvest_task = PythonOperator(
    task_id='extract_ainvest_news',
    python_callable=extract_ainvest_news,
    dag=dag,
)

extract_bitcoin_task = PythonOperator(
    task_id='extract_bitcoin_news',
    python_callable=extract_bitcoin_news,
    dag=dag,
)

check_stats_task = PythonOperator(
    task_id='check_database_stats',
    python_callable=check_database_stats,
    dag=dag,
)

# Define task dependencies
extract_ainvest_task >> extract_bitcoin_task >> check_stats_task 