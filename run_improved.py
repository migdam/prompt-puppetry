#!/usr/bin/env python3
import sys
import asyncio
import os
from openai import AsyncOpenAI, RateLimitError, APIError
from pathlib import Path
import sqlite3
from contextlib import contextmanager

# Constants
DEFAULT_DB = Path(__file__).with_name("puppetry.db")

# Enhanced strategies with more options for scientific topics
STRATEGIES = [
    # Original strategies
    ("rubric_guided",
     "Use the rubric below to teach {topic}:\n"
     "• Definition\n• Core mechanism\n• Real-world analogy\n• 2 FAQ & answers\n"
     "Respond in that exact order."),
    ("two_step_self_check",
     "Please answer in this format:\n"
     "STEP 1 – Draft: explanation of {topic}.\n"
     "STEP 2 – Critique and revise."),
    ("role_assignment",
     "You are a top professor. Explain {topic} to a 12-year-old."),
    
    # New strategies
    ("compare_models",
     "Compare classical computing and {topic} using a table format. Include pros, cons, and use cases."),
    ("step_by_step",
     "Explain {topic} in 5 clear steps, starting with the simplest concepts and building up."),
    ("history_context",
     "Provide a brief history of {topic}, then explain the key principles that make it work."),
    ("visual_analogy", 
     "Explain {topic} using a detailed visual analogy that a non-expert would understand."),
    ("contrasting_approaches", 
     "Compare and contrast the most important approaches within {topic}. Highlight strengths and limitations."),
]

# Enhanced KeywordScorer with quantum computing specific terms
class KeywordScorer:
    positive = {
        "in summary", "key idea", "fundamental", "for example", 
        "quantum", "superposition", "entanglement", "qubits",
        "to summarize", "importantly", "in essence", "notably",
        "quantum mechanics", "physics", "wave function",
        "therefore", "quantum states", "measurement", "computation",
        "analogous to", "coherence", "algorithm", "specifically",
        "ultimately", "unlike classical", "principle"
    }
    negative = {
        "i don't know", "cannot", "as an ai", "as an assistant", 
        "I'm sorry", "I don't have", "I cannot", "I can't provide", 
        "I apologize"
    }
    
    # Weight factors for certain terms
    weight_factors = {
        "superposition": 2.0,
        "entanglement": 2.0, 
        "qubits": 2.0,
        "quantum": 1.5,
        "algorithm": 1.5
    }
    
    def __call__(self, text: str) -> float:
        t = text.lower()
        score = 0.0
        
        # Base score from length (increased weight)
        score += 0.005 * len(text)
        
        # Bonus for positive signals with varying weights
        for term in self.positive:
            count = t.count(term)
            if count > 0:
                weight = self.weight_factors.get(term, 1.0)
                score += count * weight
        
        # Higher penalty for negative signals
        score -= 3 * sum(t.count(k) for k in self.negative)
        
        return round(score, 2)

# Database functions
def init_db(path):
    schema = """
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS run (
        id INTEGER PRIMARY KEY,
        strategy TEXT,
        attempt INTEGER,
        prompt TEXT,
        response TEXT,
        score REAL,
        topic TEXT,
        model TEXT,
        ts DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE INDEX IF NOT EXISTS idx_run_topic ON run(topic);
    """
    with sqlite3.connect(path) as con:
        con.executescript(schema)

@contextmanager
def open_db(path):
    init_db(path)
    con = sqlite3.connect(path)
    try:
        yield con
    finally:
        con.commit()
        con.close()

def log_run(con, **row):
    con.execute(
        "INSERT INTO run (strategy, attempt, prompt, response, score, topic, model) "
        "VALUES (:strategy, :attempt, :prompt, :response, :score, :topic, :model)",
        row
    )

# Database analysis function
def analyze_results(topic, db_path=DEFAULT_DB):
    with sqlite3.connect(db_path) as con:
        # Get top scoring results
        print(f"\n=== TOP SCORING RESULTS FOR '{topic}' ===")
        results = con.execute(
            "SELECT strategy, score, substr(response, 1, 150) FROM run "
            "WHERE topic = ? ORDER BY score DESC LIMIT 5",
            (topic,)
        ).fetchall()
        
        if not results:
            print("No results found for this topic.")
            return
            
        for i, (strategy, score, response_preview) in enumerate(results, 1):
            print(f"{i}. Strategy: {strategy}")
            print(f"   Score: {score}")
            print(f"   Preview: {response_preview}...")
            print()
            
        # Check for negative signals
        print("=== NEGATIVE SIGNALS CHECK ===")
        negative_terms = ["cannot", "as an ai", "I'm sorry", "I don't"]
        query = " OR ".join([f"response LIKE '%{term}%'" for term in negative_terms])
        negative_results = con.execute(
            f"SELECT strategy, score FROM run WHERE topic = ? AND ({query}) "
            f"ORDER BY score DESC LIMIT 3",
            (topic,)
        ).fetchall()
        
        if negative_results:
            print("Found responses with potentially penalizing language:")
            for strategy, score in negative_results:
                print(f"Strategy '{strategy}' with score {score}")
        else:
            print("No significant negative signals found.")

# Helper functions
def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            zshrc = Path.home() / ".zshrc"
            if zshrc.exists():
                text = zshrc.read_text()
                import re
                key_re = re.compile(r"^\s*export\s+OPENAI_API_KEY=['\"]*([^'\"]+)['\"]*", re.M)
                match = key_re.search(text)
                if match:
                    key = match.group(1).strip()
        except:
            pass
    
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found. Add to environment or ~/.zshrc")
    return key

def mutate(base, topic, var):
    extras = ["", " Add examples.", " Include visuals.", " Use analogies."]
    return (base + extras[var]).format(topic=topic)

async def ask(client, model, prompt, temperature=0.7, max_tokens=500):
    delay = 1
    print(f"Sending prompt to {model}...")
    while True:
        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return rsp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            print(f"API error: {e}. Retrying in {delay}s...")
            if delay > 32: 
                print("Too many retries, returning error.")
                return "[ERROR]"
            await asyncio.sleep(delay)
            delay *= 2

# Main puppeteer function with lowered threshold and improved model selection
async def puppeteer(topic, goal=3.5, use_premium_model=True):
    print(f"Running Enhanced Prompt Puppetry for topic: {topic}")
    print(f"Target score threshold: {goal}")
    
    scorer = KeywordScorer()
    key = get_api_key()
    
    # Model selection (using more capable models)
    premium_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    cheaper_model = "gpt-3.5-turbo-0125"
    
    active_model = premium_model if use_premium_model else cheaper_model
    print(f"Using model: {active_model}")
    
    db_path = DEFAULT_DB
    temperature = 0.7
    max_tokens = 600  # Increased max tokens
    
    init_db(db_path)
    with open_db(db_path) as db:
        client = AsyncOpenAI(api_key=key)
        
        for strat_name, base in STRATEGIES:
            print(f"\nTesting strategy: {strat_name}")
            prompts = [mutate(base, topic, i) for i in range(3)]
            
            async def run_variant(p, idx):
                print(f"  Variant {idx+1}...")
                r = await ask(client, active_model, p, temperature, max_tokens)
                score = scorer(r)
                log_run(db, strategy=strat_name, attempt=idx, prompt=p, response=r,
                        score=score, topic=topic, model=active_model)
                print(f"  Variant {idx+1} score: {score}")
                
                if score >= goal:
                    raise asyncio.CancelledError
                return score
            
            try:
                tasks = [asyncio.create_task(run_variant(p, i)) for i, p in enumerate(prompts)]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            except asyncio.CancelledError:
                print(f"\n✓ Success via {strat_name}")
                # Print the successful prompt and response
                with sqlite3.connect(db_path) as con:
                    result = con.execute(
                        "SELECT prompt, response, score FROM run WHERE topic = ? AND strategy = ? "
                        "ORDER BY score DESC LIMIT 1",
                        (topic, strat_name)
                    ).fetchone()
                    if result:
                        prompt, response, score = result
                        print(f"\nSuccessful prompt (Score: {score}):")
                        print("-" * 40)
                        print(prompt)
                        print("\nResponse:")
                        print("-" * 40)
                        print(response)
                return
                
        print("\n✗ No prompt succeeded in reaching the threshold score.")
        
    # After all strategies are tested, analyze the results
    print("\nAnalyzing results from the database...")
    analyze_results(topic, db_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_improved.py \"topic\" [--premium=false] [--goal=3.5]")
        sys.exit(1)
    
    topic = sys.argv[1]
    
    # Parse optional args
    premium = True
    goal = 3.5
    
    for arg in sys.argv[2:]:
        if arg.startswith("--premium="):
            premium_str = arg.split("=")[1].lower()
            premium = premium_str not in ("false", "0", "no")
        elif arg.startswith("--goal="):
            try:
                goal = float(arg.split("=")[1])
            except ValueError:
                print(f"Invalid goal value: {arg}. Using default goal of 3.5")
    
    asyncio.run(puppeteer(topic, goal=goal, use_premium_model=premium))