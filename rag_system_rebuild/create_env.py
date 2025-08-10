#!/usr/bin/env python3
"""
Quick script to create .env file with Langfuse credentials.
Run this script and it will prompt you for your credentials.
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with Langfuse credentials."""
    print("🔧 Creating .env file for Langfuse")
    print("=" * 40)
    
    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Aborted. .env file not modified.")
            return False
    
    # Get credentials from user
    print("\nPlease enter your Langfuse credentials:")
    public_key = input("LANGFUSE_PUBLIC_KEY: ").strip()
    secret_key = input("LANGFUSE_SECRET_KEY: ").strip()
    host = input("LANGFUSE_HOST (press Enter for default): ").strip()
    
    if not host:
        host = "https://cloud.langfuse.com"
    
    if not public_key or not secret_key:
        print("❌ Public key and secret key are required!")
        return False
    
    # Create .env content
    env_content = f"""# Langfuse Configuration
LANGFUSE_PUBLIC_KEY={public_key}
LANGFUSE_SECRET_KEY={secret_key}
LANGFUSE_HOST={host}
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"\n✅ .env file created successfully!")
        print(f"   File: {env_file.absolute()}")
        print(f"   Public key: {public_key[:8]}...")
        print(f"   Secret key: {secret_key[:8]}...")
        print(f"   Host: {host}")
        
        print("\n🎉 You can now run your RAG system with Langfuse!")
        print("   Test the configuration with: python test_langfuse_config.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("config.py").exists():
        print("❌ Please run this script from the rag_system_rebuild directory")
        exit(1)
    
    create_env_file() 