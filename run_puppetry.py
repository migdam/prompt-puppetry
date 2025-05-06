#!/usr/bin/env python3
import sys
import asyncio
from puppeteer_fast import puppeteer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_puppetry.py \"topic\"")
        sys.exit(1)
    
    topic = sys.argv[1]
    print(f"Running Prompt Puppetry for topic: {topic}")
    asyncio.run(puppeteer(topic))
