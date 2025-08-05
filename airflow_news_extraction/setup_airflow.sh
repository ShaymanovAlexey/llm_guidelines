#!/bin/bash

# Airflow Setup Script for News Extraction Pipeline

echo "🚀 Setting up Airflow for News Extraction Pipeline"
echo "=================================================="

# Set Airflow home directory
export AIRFLOW_HOME=/home/alex/Documents/project_for_llm/airflow_news_extraction

# Create necessary directories
mkdir -p $AIRFLOW_HOME/{dags,logs,plugins,scripts}

# Install requirements
echo "📦 Installing Airflow requirements..."
pip install -r requirements.txt

# Initialize Airflow database
echo "🗄️  Initializing Airflow database..."
airflow db init

# Create admin user
echo "👤 Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Airflow webserver (in background)
echo "🌐 Starting Airflow webserver..."
airflow webserver --port 8080 --daemon

# Start Airflow scheduler (in background)
echo "⏰ Starting Airflow scheduler..."
airflow scheduler --daemon

echo ""
echo "✅ Airflow setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Access Airflow UI at: http://localhost:8080"
echo "2. Login with: admin / admin"
echo "3. Enable the 'news_extraction_pipeline' DAG"
echo "4. The pipeline will run every 6 hours automatically"
echo ""
echo "🔧 Useful commands:"
echo "• Stop Airflow: airflow webserver stop && airflow scheduler stop"
echo "• View logs: tail -f $AIRFLOW_HOME/logs/dag_id/task_id/run_id.log"
echo "• Test DAG: airflow dags test news_extraction_pipeline $(date +%Y-%m-%d)"
echo "" 