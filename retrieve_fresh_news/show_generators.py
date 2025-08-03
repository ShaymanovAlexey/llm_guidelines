#!/usr/bin/env python3
"""
Script to demonstrate and test different summary generators.
"""

import asyncio
import sys
import os

# Add the rag_system_rebuild path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))

from summary_generator import generate_summary, SummaryGeneratorFactory
from config import SUMMARY_CONFIG, print_available_generators

async def test_summary_generators():
    """Test all available summary generators with sample content."""
    
    # Sample content for testing
    sample_content = """Artificial Intelligence (AI) has emerged as a transformative force in the investment landscape, revolutionizing how financial institutions analyze markets, manage portfolios, and make trading decisions. Machine learning algorithms can process vast amounts of data in real-time, identifying patterns and opportunities that human analysts might miss. Hedge funds and investment firms are increasingly adopting AI-powered tools for quantitative analysis, risk assessment, and automated trading strategies. The technology enables more sophisticated market predictions and helps reduce human bias in investment decisions. However, the rapid adoption of AI in finance also raises concerns about market stability, regulatory oversight, and the potential for algorithmic trading to amplify market volatility."""
    
    print("Testing Summary Generators")
    print("=" * 50)
    print(f"Sample content length: {len(sample_content)} characters")
    print()
    
    # Test each generator
    generators = ['simple', 'ollama', 'huggingface']
    
    for gen_type in generators:
        print(f"Testing {gen_type.upper()} generator:")
        print("-" * 30)
        
        try:
            summary = await generate_summary(
                sample_content, 
                max_length=150, 
                generator_type=gen_type
            )
            print(f"Summary: {summary}")
            print(f"Length: {len(summary)} characters")
        except Exception as e:
            print(f"Error: {e}")
        
        print()

async def main():
    """Main function to run the demonstration."""
    print("Summary Generator Demonstration")
    print("=" * 50)
    print()
    
    # Show available generators
    print_available_generators()
    print()
    
    # Test generators
    await test_summary_generators()
    
    print("\nTo change the default generator, edit the 'generator_type' in config.py")
    print("Available options: 'simple', 'ollama', 'huggingface'")

if __name__ == "__main__":
    asyncio.run(main()) 