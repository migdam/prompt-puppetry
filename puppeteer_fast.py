import asyncio, sys
from openai import AsyncOpenAI, RateLimitError, APIError
from config import get_cli_args, get_api_key
from db import open_db, log_run, init_db
from scorer import KeywordScorer

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

def mutate(base, topic, var):
    extras = ["", " Add examples.", " Include visuals.", " Use analogies."]
    return (base + extras[var]).format(topic=topic)

async def ask(client, model, prompt, args):
    delay = 1
    while True:
        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return rsp.choices[0].message.content
        except (RateLimitError, APIError):
            if delay > 32: return "[ERROR]"
            await asyncio.sleep(delay)
            delay *= 2

async def puppeteer(topic, goal=6.0):
    args = get_cli_args()
    scorer = KeywordScorer()
    key = get_api_key()
    cheap_model, premium_model = "gpt-3.5-turbo-0125", args.model
    init_db(args.db)
    with open_db(args.db) as db:
        client = AsyncOpenAI(api_key=key)
        for strat_name, base in STRATEGIES:
            prompts = [mutate(base, topic, i) for i in range(3)]
            async def run_variant(p):
                r = await ask(client, cheap_model, p, args)
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
        print("Usage: puppeteer_fast.py <topic>")
        sys.exit(1)
    asyncio.run(puppeteer(sys.argv[1]))