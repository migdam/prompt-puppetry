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
STRATEGIES = [
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
]

# KeywordScorer class
class KeywordScorer:
    positive = {"in summary", "key idea", "fundamental", "for example"}
    negative = {"i don't know", "cannot", "as an ai"}
    
    def __call__(self, text: str) -> float:
        t = text.lower()
        score = 0.0
        score += 0.004 * len(text)
        score += sum(t.count(k) for k in self.positive)
        score -= 2 * sum(t.count(k) for k in self.negative)
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
    while True:
        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return rsp.choices[0].message.content
        except (RateLimitError, APIError):
            if delay > 32: return "[ERROR]"
            await asyncio.sleep(delay)
            delay *= 2

# Main puppeteer function
async def puppeteer(topic, goal=6.0):
    print(f"Running Prompt Puppetry for topic: {topic}")
    
    scorer = KeywordScorer()
    key = get_api_key()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    cheap_model = "gpt-3.5-turbo-0125"
    db_path = DEFAULT_DB
    temperature = 0.7
    max_tokens = 500
    
    init_db(db_path)
    with open_db(db_path) as db:
        client = AsyncOpenAI(api_key=key)
        for strat_name, base in STRATEGIES:
            prompts = [mutate(base, topic, i) for i in range(3)]
            async def run_variant(p):
                r = await ask(client, cheap_model, p, temperature, max_tokens)
                score = scorer(r)
                log_run(db, strategy=strat_name, attempt=0, prompt=p, response=r,
                        score=score, topic=topic, model=cheap_model)
                if score >= goal:
                    raise asyncio.CancelledError
                return score
            try:
                await asyncio.wait([asyncio.create_task(run_variant(p)) for p in prompts],
                                   return_when=asyncio.FIRST_EXCEPTION)
            except asyncio.CancelledError:
                print(f"✓ Success via {strat_name}")
                return
        print("✗ No prompt succeeded.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py \"topic\"")
        sys.exit(1)
    
    topic = sys.argv[1]
    asyncio.run(puppeteer(topic))