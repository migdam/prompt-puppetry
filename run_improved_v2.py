#!/usr/bin/env python3
import sys
import asyncio
import os
import json
import time
import re
from datetime import datetime
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

# Fixed analyze_results function
def analyze_results(topic, db_path=DEFAULT_DB, limit=5):
    results = {}
    
    with sqlite3.connect(db_path) as con:
        # Get top scoring results
        print(f"\n=== TOP SCORING RESULTS FOR '{topic}' ===")
        top_results = con.execute(
            "SELECT strategy, score, substr(response, 1, 150) FROM run "
            "WHERE topic = ? ORDER BY score DESC LIMIT ?",
            (topic, limit)
        ).fetchall()
        
        results["top_scores"] = []
        
        if not top_results:
            print("No results found for this topic.")
            results["top_scores"] = []
            return results
            
        for i, (strategy, score, response_preview) in enumerate(top_results, 1):
            print(f"{i}. Strategy: {strategy}")
            print(f"   Score: {score}")
            print(f"   Preview: {response_preview}...")
            print()
            results["top_scores"].append({
                "strategy": strategy,
                "score": score,
                "preview": response_preview
            })
            
        # Check for negative signals - FIXED QUERY
        print("=== NEGATIVE SIGNALS CHECK ===")
        negative_terms = ["cannot", "as an ai", "sorry", "dont"]
        
        # Fixed SQL query construction
        query = "SELECT strategy, score FROM run WHERE topic = ? AND ("
        query += " OR ".join([f"response LIKE ?" for _ in negative_terms])
        query += ") ORDER BY score DESC LIMIT 3"
        
        # Prepare parameters
        params = [topic] + [f"%{term}%" for term in negative_terms]
        
        negative_results = con.execute(query, params).fetchall()
        results["negative_signals"] = []
        
        if negative_results:
            print("Found responses with potentially penalizing language:")
            for strategy, score in negative_results:
                print(f"Strategy '{strategy}' with score {score}")
                results["negative_signals"].append({
                    "strategy": strategy,
                    "score": score
                })
        else:
            print("No significant negative signals found.")
            
        # Get strategy statistics
        results["strategy_stats"] = []
        for strategy_name, _ in STRATEGIES:
            stats = con.execute(
                "SELECT COUNT(*), AVG(score), MAX(score), MIN(score) FROM run "
                "WHERE topic = ? AND strategy = ?",
                (topic, strategy_name)
            ).fetchone()
            
            if stats[0] > 0:  # If there are results
                results["strategy_stats"].append({
                    "strategy": strategy_name,
                    "count": stats[0],
                    "avg_score": round(stats[1], 2) if stats[1] else 0,
                    "max_score": round(stats[2], 2) if stats[2] else 0,
                    "min_score": round(stats[3], 2) if stats[3] else 0
                })
        
        # Get distribution of scores
        results["score_distribution"] = []
        ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), 
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 1000)]
        
        for low, high in ranges:
            count = con.execute(
                "SELECT COUNT(*) FROM run WHERE topic = ? AND score >= ? AND score < ?",
                (topic, low, high)
            ).fetchone()[0]
            
            if count > 0:
                results["score_distribution"].append({
                    "range": f"{low}-{high}",
                    "count": count
                })
        
        # Get best prompts for each strategy
        results["best_prompts"] = []
        for strategy_name, _ in STRATEGIES:
            best = con.execute(
                "SELECT prompt, score FROM run WHERE topic = ? AND strategy = ? "
                "ORDER BY score DESC LIMIT 1",
                (topic, strategy_name)
            ).fetchone()
            
            if best:
                results["best_prompts"].append({
                    "strategy": strategy_name,
                    "prompt": best[0],
                    "score": best[1]
                })
                
    return results

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

# Fixed success handling in puppeteer function
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
    success_found = False
    successful_strategy = None
    successful_prompt = None
    successful_response = None
    successful_score = 0
    
    with open_db(db_path) as db:
        client = AsyncOpenAI(api_key=key)
        
        for strat_name, base in STRATEGIES:
            if success_found:
                break
                
            print(f"\nTesting strategy: {strat_name}")
            prompts = [mutate(base, topic, i) for i in range(3)]
            
            results = []
            for idx, p in enumerate(prompts):
                print(f"  Variant {idx+1}...")
                r = await ask(client, active_model, p, temperature, max_tokens)
                score = scorer(r)
                log_run(db, strategy=strat_name, attempt=idx, prompt=p, response=r,
                        score=score, topic=topic, model=active_model)
                print(f"  Variant {idx+1} score: {score}")
                
                results.append((p, r, score))
                
                # Check for success
                if score >= goal:
                    success_found = True
                    successful_strategy = strat_name
                    successful_prompt = p
                    successful_response = r
                    successful_score = score
                    break
        
        if success_found:
            print(f"\n✓ Success via {successful_strategy}")
            print(f"\nSuccessful prompt (Score: {successful_score}):")
            print("-" * 40)
            print(successful_prompt)
            print("\nResponse:")
            print("-" * 40)
            print(successful_response)
        else:        
            print("\n✗ No prompt succeeded in reaching the threshold score.")
        
    # After all strategies are tested, analyze the results
    print("\nAnalyzing results from the database...")
    analysis_results = analyze_results(topic, db_path)
    
    # Generate HTML report
    generate_html_report(topic, analysis_results, db_path)
    
    return success_found, analysis_results

# Generate AI-powered insights from results
async def generate_ai_insights(topic, results, model="gpt-4o-mini"):
    client = AsyncOpenAI(api_key=get_api_key())
    
    # Create a prompt for AI to analyze the results
    prompt = f"""
    Analyze these prompt engineering results for the topic "{topic}":
    
    Top scoring strategies: {json.dumps(results['strategy_stats'], indent=2)}
    Score distribution: {json.dumps(results['score_distribution'], indent=2)}
    
    Please provide insights on:
    1. Which prompt strategies work best for this topic and why
    2. Common patterns in high-scoring prompts
    3. Recommendations for creating effective prompts on this topic
    4. Any surprising findings from the data
    
    Format your response as JSON with the following structure:
    {{
      "best_strategies": [list of strategies with explanations],
      "patterns": [list of effective patterns],
      "recommendations": [list of actionable recommendations],
      "surprises": [list of unexpected findings]
    }}
    
    Include at least 3 items in each category. Keep your response concise and specific.
    """
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        
        insight_text = response.choices[0].message.content
        
        # Extract JSON from response
        json_match = re.search(r'({[\s\S]*})', insight_text)
        if json_match:
            try:
                insights = json.loads(json_match.group(1))
                return insights
            except json.JSONDecodeError:
                return {"error": "Could not parse AI insights"}
        else:
            return {"error": "No JSON found in AI response"}
    
    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return {"error": f"Failed to generate insights: {str(e)}"}

# Generate HTML report
def generate_html_report(topic, results, db_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{topic.replace(' ', '_')}_{timestamp}.html"
    
    # Get successful examples if any
    best_examples = []
    with sqlite3.connect(db_path) as con:
        examples = con.execute(
            "SELECT strategy, prompt, response, score FROM run "
            "WHERE topic = ? ORDER BY score DESC LIMIT 3",
            (topic,)
        ).fetchall()
        
        for strategy, prompt, response, score in examples:
            best_examples.append({
                "strategy": strategy,
                "prompt": prompt,
                "response": response[:1000] + ("..." if len(response) > 1000 else ""),
                "score": score
            })
    
    # Get AI insights asynchronously
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        insights = loop.run_until_complete(generate_ai_insights(topic, results))
    else:
        # If we're already in an event loop, just set a placeholder
        insights = {"note": "AI insights not available in nested event loop"}
    
    # Prepare data for charts
    strategy_names = [stat["strategy"] for stat in results["strategy_stats"]]
    avg_scores = [stat["avg_score"] for stat in results["strategy_stats"]]
    max_scores = [stat["max_score"] for stat in results["strategy_stats"]]
    
    score_ranges = [item["range"] for item in results["score_distribution"]]
    score_counts = [item["count"] for item in results["score_distribution"]]
    
    # Sort strategies by max score for better visualization
    sorted_indices = sorted(range(len(max_scores)), key=lambda i: max_scores[i], reverse=True)
    strategy_names = [strategy_names[i] for i in sorted_indices]
    avg_scores = [avg_scores[i] for i in sorted_indices]
    max_scores = [max_scores[i] for i in sorted_indices]
    
    # Create HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Puppetry Results: {topic}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            flex: 1;
            min-width: 300px;
        }}
        .chart-container {{
            width: 100%;
            height: 400px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stat-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            flex: 1;
            margin: 0 10px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .example {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .example h3 {{
            color: #3498db;
            margin-top: 0;
        }}
        .score {{
            font-size: 18px;
            font-weight: bold;
            color: #e67e22;
        }}
        pre {{
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            border-left: 3px solid #3498db;
        }}
        .insight-item {{
            background: #fff;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 3px solid #9b59b6;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .tab {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 8px 8px 0 0;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #3498db;
            color: white;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 8px 8px;
            animation: fadeEffect 1s;
            background-color: white;
        }}
        @keyframes fadeEffect {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}
    </style>
</head>
<body>
    <h1>Prompt Puppetry Results: {topic}</h1>
    
    <div class="stats">
        <div class="stat-item">
            <div class="stat-value">{len(results["strategy_stats"])}</div>
            <div>Strategies Tested</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{sum(stat["count"] for stat in results["strategy_stats"])}</div>
            <div>Total Prompts</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{max(stat["max_score"] for stat in results["strategy_stats"]) if results["strategy_stats"] else 0}</div>
            <div>Highest Score</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{round(sum(stat["avg_score"] * stat["count"] for stat in results["strategy_stats"]) / sum(stat["count"] for stat in results["strategy_stats"]) if sum(stat["count"] for stat in results["strategy_stats"]) > 0 else 0, 2)}</div>
            <div>Average Score</div>
        </div>
    </div>
    
    <div class="tab">
        <button class="tablinks active" onclick="openTab(event, 'Overview')">Overview</button>
        <button class="tablinks" onclick="openTab(event, 'Examples')">Best Examples</button>
        <button class="tablinks" onclick="openTab(event, 'Insights')">AI Insights</button>
        <button class="tablinks" onclick="openTab(event, 'Data')">Raw Data</button>
    </div>
    
    <div id="Overview" class="tabcontent" style="display: block;">
        <div class="container">
            <div class="card">
                <h2>Strategy Performance</h2>
                <div class="chart-container">
                    <canvas id="strategyChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Score Distribution</h2>
                <div class="chart-container">
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Strategy Rankings</h2>
                <table style="width:100%; border-collapse: collapse;">
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Strategy</th>
                        <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Avg Score</th>
                        <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Max Score</th>
                        <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Count</th>
                    </tr>
                    {"".join(f'<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stat["strategy"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{stat["avg_score"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{stat["max_score"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{stat["count"]}</td></tr>' for stat in sorted(results["strategy_stats"], key=lambda x: x["max_score"], reverse=True))}
                </table>
            </div>
        </div>
    </div>
    
    <div id="Examples" class="tabcontent">
        <h2>Top Performing Examples</h2>
        
        {"".join(f'''
        <div class="example">
            <h3>{example["strategy"]}</h3>
            <div class="score">Score: {example["score"]}</div>
            <h4>Prompt:</h4>
            <pre>{example["prompt"]}</pre>
            <h4>Response:</h4>
            <pre>{example["response"]}</pre>
        </div>
        ''' for example in best_examples)}
    </div>
    
    <div id="Insights" class="tabcontent">
        <h2>AI-Generated Insights</h2>
        
        <div class="container">
            <div class="card">
                <h3>Best Performing Strategies</h3>
                {"".join(f'<div class="insight-item"><p>{insight}</p></div>' for insight in insights.get("best_strategies", ["AI insights not available"]))}
            </div>
            
            <div class="card">
                <h3>Effective Patterns</h3>
                {"".join(f'<div class="insight-item"><p>{pattern}</p></div>' for pattern in insights.get("patterns", ["AI insights not available"]))}
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h3>Recommendations</h3>
                {"".join(f'<div class="insight-item"><p>{rec}</p></div>' for rec in insights.get("recommendations", ["AI insights not available"]))}
            </div>
            
            <div class="card">
                <h3>Surprising Findings</h3>
                {"".join(f'<div class="insight-item"><p>{surprise}</p></div>' for surprise in insights.get("surprises", ["AI insights not available"]))}
            </div>
        </div>
    </div>
    
    <div id="Data" class="tabcontent">
        <h2>Raw Data</h2>
        <pre style="max-height: 500px; overflow-y: auto;">{json.dumps(results, indent=2)}</pre>
    </div>
    
    <div class="footer">
        <p>Generated by Prompt Puppetry on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <script>
        // Tab functionality
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Strategy performance chart
        const strategyCtx = document.getElementById('strategyChart').getContext('2d');
        const strategyChart = new Chart(strategyCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(strategy_names)},
                datasets: [
                    {{
                        label: 'Average Score',
                        data: {json.dumps(avg_scores)},
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Max Score',
                        data: {json.dumps(max_scores)},
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Score'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Strategy'
                        }}
                    }}
                }}
            }}
        }});
        
        // Score distribution chart
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        const distributionChart = new Chart(distributionCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(score_ranges)},
                datasets: [
                    {{
                        label: 'Number of Prompts',
                        data: {json.dumps(score_counts)},
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Count'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Score Range'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write HTML to file
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"\nHTML report generated: {report_filename}")
    return report_filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_improved_v2.py \"topic\" [--premium=false] [--goal=3.5]")
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