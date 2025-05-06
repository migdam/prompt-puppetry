# Prompt Puppetry

A high-efficiency prompt engineering and testing tool designed to find the most effective prompt strategies for any given topic using OpenAI models.

## About

Prompt Puppetry automates the process of finding the best prompt structures for getting high-quality responses from large language models. It tests multiple prompt strategies in parallel, scores the responses, and identifies the most effective approaches.

## Features

- **Async Batch Processing**: Test multiple prompt variants simultaneously
- **Smart Scoring System**: Evaluate responses based on content quality signals
- **Early Success Detection**: Stop testing when a high-scoring prompt is found
- **Multiple Strategy Testing**: Compare different prompting approaches
- **SQLite Logging**: Store all results for later analysis
- **Flexible API Key Detection**: Load keys from environment variables, .env file, or ~/.zshrc

## Installation

### Using pip (Python 3.9+ recommended)

```bash
# Clone the repository
git clone https://github.com/migdam/prompt-puppetry.git
cd prompt-puppetry

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/migdam/prompt-puppetry.git
cd prompt-puppetry

# Create and activate conda environment
conda env create -f environment.yml
conda activate prompt-puppetry
```

## Setup

1. Ensure you have an OpenAI API key:
   - Set as environment variable: `export OPENAI_API_KEY=your-key-here`
   - Add to .env file: `OPENAI_API_KEY=your-key-here`
   - Add to your ~/.zshrc: `export OPENAI_API_KEY=your-key-here`

2. Optional: Configure your preferred model:
   - Set environment variable: `export OPENAI_MODEL=gpt-4o`

## Usage

### Basic Usage

```bash
python puppeteer_fast.py "quantum computing"
```

### Advanced Options

```bash
python puppeteer_fast.py "machine learning" --model "gpt-4o" --temperature 0.5 --max-tokens 750
```

## How It Works

1. **Strategy Selection**: The tool loads prompt templates from `strategies.json`
2. **Variation Generation**: Each strategy is modified with additional instructions
3. **Async Execution**: Modified prompts are tested in parallel
4. **Response Scoring**: Each response is scored based on:
   - Text length
   - Presence of positive signal phrases ("in summary", "key idea", etc.)
   - Absence of negative signal phrases ("I don't know", "as an AI", etc.)
5. **Database Logging**: All results are stored in SQLite for analysis
6. **Early Termination**: Testing stops when a response exceeds the score threshold

## Troubleshooting

If you encounter any issues with the command-line arguments, make sure to:

1. **Put multi-word topics in quotes**: Always wrap your topic in quotes if it contains spaces, like `"quantum computing"`.
2. **Check your API key**: Ensure your OpenAI API key is correctly set up in one of the supported methods.
3. **Database permissions**: Make sure you have write permissions in the directory where the database is being created.
4. **Model availability**: If you specify a custom model, ensure it's available in your OpenAI account.

## Project Structure

```
.
├── README.md           # This documentation
├── config.py           # Configuration and API key management
├── db.py               # Database operations
├── environment.yml     # Conda environment configuration
├── puppeteer_fast.py   # Main script with async evaluation
├── requirements.txt    # Python dependencies
├── scorer.py           # Response evaluation logic 
├── strategies.json     # Prompt template definitions
└── test_puppeteer.py   # Unit tests
```

## Custom Prompt Strategies

Modify `strategies.json` to add your own prompt templates:

```json
{
  "my_new_strategy": "Explain {topic} using these steps: 1) Define it 2) Give examples 3) Explain applications"
}
```

The `{topic}` placeholder will be replaced with your query topic.

## Customizing the Scorer

Edit `scorer.py` to modify how responses are evaluated:

```python
class KeywordScorer:
    # Add your own positive and negative signal phrases
    positive = {"in summary", "key idea", "fundamental", "for example", "consequently"}
    negative = {"i don't know", "cannot", "as an ai", "as an assistant"}
    
    # Modify scoring algorithm
    def __call__(self, text: str) -> float:
        t = text.lower()
        score = 0.0
        # Base score from length (0.004 points per character)
        score += 0.004 * len(text)
        # Bonus for positive signals
        score += sum(t.count(k) for k in self.positive)
        # Penalty for negative signals
        score -= 2 * sum(t.count(k) for k in self.negative)
        return round(score, 2)
```

## Testing

```bash
python -m unittest test_puppeteer.py
```

## License

MIT License

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.