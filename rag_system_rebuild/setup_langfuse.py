#!/usr/bin/env python3
"""
Setup script for Langfuse configuration in RAG System Rebuild.
This script helps you configure your Langfuse credentials and test the connection.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables for Langfuse."""
    print("🔧 Langfuse Setup for RAG System Rebuild")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create .env file
    print("📝 Creating .env file...")
    
    # Get credentials from user
    public_key = input("Enter your LANGFUSE_PUBLIC_KEY: ").strip()
    secret_key = input("Enter your LANGFUSE_SECRET_KEY: ").strip()
    host = input("Enter your LANGFUSE_HOST (or press Enter for default): ").strip()
    
    if not host:
        host = "https://cloud.langfuse.com"
    
    if not public_key or not secret_key:
        print("❌ Public key and secret key are required!")
        return False
    
    # Write .env file
    env_content = f"""# Langfuse Configuration
LANGFUSE_PUBLIC_KEY={public_key}
LANGFUSE_SECRET_KEY={secret_key}
LANGFUSE_HOST={host}
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_langfuse_connection():
    """Test the Langfuse connection."""
    print("\n🧪 Testing Langfuse connection...")
    
    try:
        # Import after setting up environment
        from config import get_config
        from langfuse_integration import LangfuseManager
        
        config = get_config()
        langfuse_manager = LangfuseManager(config.langfuse)
        
        if langfuse_manager.is_enabled():
            print("✅ Langfuse connection successful!")
            print(f"   Project: {config.langfuse.project_name}")
            print(f"   Host: {config.langfuse.host}")
            return True
        else:
            print("❌ Langfuse connection failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have installed the required packages:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Welcome to RAG System Rebuild Langfuse Setup!")
    
    # Check if we're in the right directory
    if not Path("config.py").exists():
        print("❌ Please run this script from the rag_system_rebuild directory")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed!")
        sys.exit(1)
    
    # Test connection
    if test_langfuse_connection():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Your .env file is now configured")
        print("2. Langfuse integration is ready to use")
        print("3. You can now run your RAG system with observability")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please check your credentials and try again")

if __name__ == "__main__":
    main() 