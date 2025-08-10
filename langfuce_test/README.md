# Langfuse + Ollama Example

This project demonstrates how to use Langfuse for LLM observability and tracing with local Ollama models.

## What is Langfuse?

Langfuse is an open-source LLM observability and analytics platform that helps you:
- **Trace** LLM applications end-to-end
- **Monitor** performance and costs
- **Debug** issues in production
- **Analyze** user interactions and model performance
- **Score** responses for quality assessment

## What is Ollama?

Ollama is a framework for running large language models locally on your machine. It provides:
- **Local inference** - no need for cloud API calls
- **Multiple models** - Llama2, Mistral, CodeLlama, and more
- **Easy setup** - simple installation and model management
- **Cost-effective** - run models without per-token costs

## Features Demonstrated

This example shows:

1. **Simple Tracing** - Basic LLM call tracing with Ollama
2. **Complex Tracing** - Multi-span RAG pipeline tracing
3. **Scoring** - Quality metrics and evaluation
4. **Model Comparison** - Comparing different Ollama models
5. **Error Handling** - Graceful error handling and tracing

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **Langfuse account** (free at [cloud.langfuse.com](https://cloud.langfuse.com))

## Installation

1. **Install Ollama:**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull some models:**
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull codellama
   ```

## Configuration

1. **Get Langfuse credentials:**
   - Go to [cloud.langfuse.com](https://cloud.langfuse.com)
   - Create a new project
   - Copy your Public Key and Secret Key

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file:**
   ```bash
   # Langfuse Configuration
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
   LANGFUSE_HOST=https://cloud.langfuse.com
   
   # Ollama Configuration
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama2
   ```

## Usage

### Run all examples:
```bash
python simple_langfuse_example.py
```

### Run specific examples:
```python
from simple_langfuse_example import LangfuseOllamaExample

example = LangfuseOllamaExample()

# Simple trace
example.create_simple_trace("What is machine learning?")

# Complex RAG trace
example.create_complex_trace("Explain quantum computing")

# Scored trace
example.create_score_trace("How does photosynthesis work?")

# Model comparison
example.create_model_comparison_trace("Write a haiku about coding")
```

## Example Output

```
🚀 Starting Langfuse + Ollama Examples
============================================================

🔍 Creating simple trace for: 'What is machine learning?'
✅ Simple trace completed in 2.34s
📝 Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...

🔍 Creating complex trace for: 'Explain the benefits of renewable energy'
✅ Complex trace completed successfully
📝 RAG Response: Renewable energy offers numerous benefits including reduced greenhouse gas emissions, energy independence...

🔍 Creating scored trace for: 'How does photosynthesis work?'
✅ Scored trace completed successfully
📝 Response: Photosynthesis is the process by which plants convert sunlight into chemical energy...

🔍 Creating model comparison trace for: 'Write a short poem about coding'
✅ Model comparison trace completed successfully
📝 llama2: In lines of code we find our way...
📝 mistral: Through algorithms we create...
📝 codellama: Functions dance in harmony...

============================================================
✅ All examples completed!
📊 Check your Langfuse dashboard to see the traces
🤖 All LLM calls were made using local Ollama models
```

## Langfuse Dashboard

After running the examples, visit your Langfuse dashboard to see:

- **Traces** - Complete execution flows
- **Spans** - Individual operations within traces
- **Generations** - LLM calls and responses
- **Scores** - Quality metrics and evaluations
- **Analytics** - Performance insights and trends

## Customization

### Add new models:
```python
# In the create_model_comparison_trace method
models_to_test = ["llama2", "mistral", "codellama", "your_custom_model"]
```

### Custom scoring:
```python
generation.score(
    name="custom_metric",
    value=0.95,
    comment="Your custom evaluation criteria"
)
```

### Add new trace types:
```python
def create_custom_trace(self, user_input: str):
    trace = self.langfuse.trace(
        name="custom-trace",
        user_id="demo-user",
        metadata={"example_type": "custom"}
    )
    
    # Add your custom logic here
    # ...
    
    trace.end()
```

## Troubleshooting

### Ollama connection issues:
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

### Langfuse connection issues:
- Verify your API keys are correct
- Check if your Langfuse project is active
- Ensure internet connectivity

### Model not found:
```bash
# Pull the required model
ollama pull llama2

# List available models
ollama list
```

## Next Steps

1. **Integrate with your application** - Add Langfuse tracing to your existing LLM app
2. **Custom scoring** - Implement domain-specific quality metrics
3. **Production monitoring** - Set up alerts and dashboards
4. **A/B testing** - Compare different models and prompts
5. **Cost optimization** - Monitor and optimize your LLM usage

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [Ollama Python Client](https://github.com/ollama/ollama-python)

## License

This example is provided as-is for educational purposes. Feel free to modify and use in your own projects. 