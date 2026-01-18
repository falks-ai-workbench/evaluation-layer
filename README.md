# Sentiment Analysis Benchmark

Automated LLM benchmark for dealership customer message sentiment analysis. Compares 5 Ollama models with 2 prompt variants against human scores.

## Features

- 5 Ollama models: Qwen2.5, Mistral, Llama3.1/3.2, GPT-OSS
- 2 prompt variants: Original vs Optimized  
- Automatic accuracy calculation vs human scores
- Top-5 ranking of best combinations
- CSV export for further analysis
- Robust error handling and progress tracking

## Example Output

=== LLM COMPARISON (all prompts) ===
gpt-oss:20b                         | optimized    = 89.00%
gpt-oss:20b                         | original     = 97.00%
llama3.1:8b-instruct-q4_K_S         | optimized    = 97.00%
llama3.1:8b-instruct-q4_K_S         | original     = 97.00%
llama3.2:latest                     | optimized    = 76.00%
llama3.2:latest                     | original     = 87.00%
ministral-3:8b-instruct-2512-q4_K_M | optimized    = 92.00%
ministral-3:8b-instruct-2512-q4_K_M | original     = 96.00%
qwen2.5:7b-instruct-q4_K_M          | optimized    = 77.00%
qwen2.5:7b-instruct-q4_K_M          | original     = 97.00%

=== TOP 5 COMBINATIONS ===
🏅 gpt-oss:20b                         | original     = 97.00%
🏅 llama3.1:8b-instruct-q4_K_S         | optimized    = 97.00%
🏅 llama3.1:8b-instruct-q4_K_S         | original     = 97.00%
🏅 qwen2.5:7b-instruct-q4_K_M          | original     = 97.00%
🏅 ministral-3:8b-instruct-2512-q4_K_M | original     = 96.00%


## Quickstart

### 1. Prerequisites
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull ministral-3b-instruct-2512-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_S
ollama pull llama3.2:latest
ollama pull gpt-oss:20b
```

### 2. Prepare Excel Data

You will need to create a sentiment.xlsx file with the following structure:

| Nr. | Kundennachricht        | score_human |
|-----|------------------------|-------------|
| 1   | Super Service!         | 9           |
| 2   | Nie wieder...          | 2           |

### 3. RUN
pip install openai pandas
python sentiment_benchmark.py

### File Structure
sentiment-benchmark/
├── sentiment.xlsx
├── sentiment_benchmark.py
├── sentiment_results.csv
├── README.md
├── .gitignore
└── .env -> Ollama do not need an API key. If you work with other providers, you can add your API key here.

### Configuation
```python
@dataclass
class Config:
    excel_path: str = 'sentiment.xlsx'
    output_csv: str = 'sentiment_results.csv'
    llms: List[str] = None

```

### CSV Output Schema
|Column               |Description              |
|---------------------|-------------------------|
|model                |Ollama model             |
|prompt               |original/optimized       |
|text                 |Original customer text   |
|sentiment            |negative/neutral/positive|
|score                |LLM score (1-10)         |
|score_accuracy_points|Accuracy vs human (0-1)  |
|score_accuracy_pct   |Accuracy in %            |


|Original           |Optimized        |
|-------------------|-----------------|
|25+ lines          |10 lines         |
|Full explanations  |Essential minimum|
|Examples + rules   |Schema + rules   |

### License
MIT License - feel free to use and modify