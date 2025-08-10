#!/usr/bin/env python3
"""
Script to show available Ollama models and help configure the RAG system
"""

import asyncio
from ollama_generator import OllamaGenerator

async def show_models():
    print("=== Available Ollama Models ===\n")
    
    # Create generator to get available models
    gen = OllamaGenerator()
    models = gen.get_available_models()
    
    print("📋 Models available in your Ollama installation:")
    for i, model in enumerate(models, 1):
        print(f"   {i:2d}. {model}")
    
    print(f"\n🎯 Total models: {len(models)}")
    
    # Test a few key models
    print("\n🧪 Testing key models:")
    
    test_models = [
        "llama3.2:latest",
        "qwen3:8b", 
        "phi4:14b",
        "deepseek-r1:14b"
    ]
    
    for model in test_models:
        if model in models:
            try:
                test_gen = OllamaGenerator(model_name=model)
                health = await test_gen.health_check()
                status = "✅" if health['status'] == 'healthy' else "❌"
                print(f"   {status} {model}: {health['status']}")
            except Exception as e:
                print(f"   ❌ {model}: Error - {e}")
        else:
            print(f"   ⚠️  {model}: Not available")
    
    print("\n💡 Recommendation:")
    print("   - For general use: llama3.2:latest (fast, good quality)")
    print("   - For coding: qwen2.5-coder:14b (specialized for code)")
    print("   - For high quality: deepseek-r1:14b (excellent reasoning)")
    
    print("\n🔧 To change the default model in main_hybrid.py:")
    print("   Edit the line: generator = OllamaGenerator(model_name='YOUR_CHOSEN_MODEL')")

if __name__ == "__main__":
    asyncio.run(show_models()) 