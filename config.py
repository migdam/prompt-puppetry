import os
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB = Path(__file__).with_name("puppetry.db")
ZSHRC = Path.home() / ".zshrc"
OPENAI_RE = re.compile(r"^\s*export\s+OPENAI_API_KEY=['\"]*([^'\"]+)['\"]*", re.M)

def get_cli_args():
    p = argparse.ArgumentParser(prog="prompt-puppetry")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=500)
    return p.parse_args()

def read_key_from_zshrc():
    try:
        text = ZSHRC.read_text()
    except FileNotFoundError:
        return None
    match = OPENAI_RE.search(text)
    return match.group(1).strip() if match else None

def get_api_key():
    if (key := os.getenv("OPENAI_API_KEY")):
        return key
    if (key := read_key_from_zshrc()):
        return key
    raise RuntimeError("OPENAI_API_KEY not found. Add to ~/.zshrc or export it.")