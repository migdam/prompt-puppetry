# Prompt Puppetry

A high-efficiency prompt engineering and testing tool designed to find the most effective prompt strategies for any given topic using OpenAI models.

## Features
- Async batch prompt evaluation
- Auto-scoring of outputs based on quality heuristics
- SQLite database logging
- Smart early-exit when a good prompt is found
- API key loaded from `.env`, `~/.zshrc`, or environment

## Installation
```bash
pip install openai python-dotenv
```

## Usage
```bash
python puppeteer_fast.py "quantum computing"
```

## Project Structure
```
.
├── config.py         # API key detection via .zshrc or .env
├── db.py             # SQLite setup and logging
├── puppeteer_fast.py # Main script with async evaluation
├── scorer.py         # Keyword-based scoring system 
├── strategies.json   # Prompt templates configuration
├── README.md         # This documentation
└── test_puppeteer.py # Unit tests
```

## How It Works

Prompt Puppetry tests different prompt strategies against language models to find the most effective approach for a given topic. It:

1. Takes a topic as input
2. Tests multiple prompt strategies defined in strategies.json
3. Evaluates responses using a keyword-based scoring system
4. Logs results to a SQLite database
5. Stops early when a successful prompt is found (above score threshold)

## Extending

Add new prompt strategies in `strategies.json` to test different approaches. Modify the `KeywordScorer` class to adjust how responses are evaluated.