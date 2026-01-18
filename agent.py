from openai import OpenAI
import json
import pandas as pd
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
from enum import Enum


# === CONFIGURATION ===
@dataclass
class Config:
    excel_path: str = 'sentiment.xlsx'
    output_csv: str = 'sentiment_results.csv'
    llms: List[str] = None
    prompt_version: str = "Prompt Version 1"

    def __post_init__(self):
        if self.llms is None:
            self.llms = [
                "qwen2.5:7b-instruct-q4_K_M",
                "ministral-3:8b-instruct-2512-q4_K_M", 
                "llama3.1:8b-instruct-q4_K_S",
                "llama3.2:latest",
                "gpt-oss:20b"
            ]


class PromptVersion(Enum):
    ORIGINAL = "original"
    OPTIMIZED = "optimized"


# === BOTH PROMPTS ===
PROMPTS = {
    PromptVersion.ORIGINAL: """You are a sentiment analysis bot for customer messages in a car dealership.
Task:
Analyze the following text for sentiment. Distinguish strictly between negative, neutral, and positive. Assign a score between 1 and 10, where 1 is very bad and 10 is outstanding (whole numbers only, no decimals).

Output format:
Always respond exclusively with a single JSON object in the following format:
{
"text": "Customer message text here...",
"sentiment": "positive",
"score": 10
}

Definitions:
text: the original customer message (unchanged).
sentiment: exactly one of "negative", "neutral", or "positive".
score: whole number between 1 and 10.

Rules:
1. Strictly adhere to the JSON format above. No additional fields, no Markdown, no explanations.
2. Always respond in German.
3. If the text cannot be meaningfully evaluated, return an empty object instead.""",
    
    PromptVersion.OPTIMIZED: """You are a sentiment analysis bot for car dealership customer messages.
Task:
Analyze the text sentiment. Distinguish strictly: negative, neutral, positive. Score 1-10 (whole numbers only).

Output format (exactly like this!):
{
  "text": "Customer message text...",
  "sentiment": "positive",
  "score": 10
}

Rules:
1. JSON object only, no explanations or Markdown.
2. Always respond in German.
3. On problems: return empty object."""
}


# === CORE LOGIC ===
class SentimentAnalyzer:
    def __init__(self, client: OpenAI):
        self.client = client

    def _extract_json(self, raw: str) -> str:
        """Extracts JSON from Markdown code blocks."""
        if '```json' in raw or '```' in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"⚠️ No JSON found: {raw[:100]}...")
                return raw
            return raw[start:end]
        return raw

    def _safe_parse(self, json_str: str, original_text: str) -> Dict:
        """Safe JSON parsing with fallback."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"⚠️ JSON error: {json_str[:100]}...")
            return {"text": original_text, "sentiment": "error", "score": 0}

    def analyze(self, model: str, text: str, human_score: float, prompt_version: PromptVersion) -> Dict:
        """Single sentiment analysis with accuracy calculation."""
        prompt = PROMPTS[prompt_version]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{prompt}\n\nCUSTOMER TEXT: {text}"},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            raw = response.choices[0].message.content.strip()
            clean_json = self._extract_json(raw)
            parsed = self._safe_parse(clean_json, text)
            
            accuracy = 1 - abs(parsed["score"] - human_score) / 10
            
            return {
                "model": model,
                "prompt": prompt_version.value,
                "text": parsed["text"],
                "sentiment": parsed["sentiment"],
                "score": parsed["score"],
                "score_accuracy_points": accuracy,
                "score_accuracy_pct": accuracy * 100
            }
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            return {
                "model": model, "prompt": prompt_version.value, "text": text,
                "sentiment": "error", "score": 0, "score_accuracy_points": 0, "score_accuracy_pct": 0
            }


def load_data(path: str) -> Tuple[pd.Series, pd.Series]:
    """Loads and cleans Excel data."""
    df = pd.read_excel(path)
    df['Kundennachricht'] = df['Kundennachricht'].str.replace('"', '', regex=False)
    df = df.drop('Nr.', axis=1)
    return df['Kundennachricht'].reset_index(drop=True), df['score_human'].reset_index(drop=True)


def print_model_comparison(df: pd.DataFrame):
    """Extended evaluation with prompt comparison."""
    print("\n=== LLM COMPARISON (all prompts) ===")
    
    # Per model + prompt
    model_prompt_accuracy = df.groupby(['model', 'prompt'])['score_accuracy_points'].mean()
    for (model, prompt), acc in model_prompt_accuracy.items():
        print(f"{model:<30} | {prompt:<12} = {acc:.2%}")
    
    # Top 5 combinations
    print("\n=== TOP 5 COMBINATIONS ===")
    top5 = df.groupby(['model', 'prompt'])['score_accuracy_points'].mean().nlargest(5)
    for (model, prompt), acc in top5.items():
        print(f"🏅 {model:<30} | {prompt:<12} = {acc:.2%}")
    
    # Summary table
    summary = df.groupby(['model', 'prompt'])['score_accuracy_points'].agg(['mean', 'count'])
    summary['mean_pct'] = (summary['mean'] * 100).round(1)
    print("\n=== FULL STATISTICS ===")
    print(summary[['mean_pct', 'count']])


# === ENTRY POINT ===
def main():
    config = Config()
    
    # Load data
    print("📊 Loading data...")
    texts, human_scores = load_data(config.excel_path)
    print(f"✅ {len(texts)} texts loaded")
    
    # Initialize client
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    analyzer = SentimentAnalyzer(client)
    
    # Run both prompts
    results = []
    for prompt_version in PromptVersion:
        print(f"\n{'='*60}")
        print(f"🔄 PROMPT: {prompt_version.value.upper()}")
        print(f"{'='*60}")
        
        for model in config.llms:
            print(f"\n--- {model} ---")
            for text, human_score in zip(texts, human_scores):
                result = analyzer.analyze(model, text, human_score, prompt_version)
                print(f"  {result['score']} ({result['sentiment']}) | Accuracy: {result['score_accuracy_pct']:.0f}%")
                results.append(result)
    
    # Save and evaluate results
    df_results = pd.DataFrame(results)
    df_results.to_csv(config.output_csv, index=False)
    print(f"\n💾 Results saved: {config.output_csv}")
    
    print_model_comparison(df_results)


if __name__ == "__main__":
    main()
