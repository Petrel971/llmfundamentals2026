"""
Compare LLMs from Anthropic and OpenAI on the same prompts, using your LLMWrapper.
"""
import csv
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.common.llmwrapper import LLMWrapper

# (provider, model)
MODELS = [
    ("anthropic", "anthropic/claude-haiku-4.5"),
    ("anthropic", "anthropic/claude-sonnet-5.5"),
    ("openai", "openai/gpt-4.1-mini"),
    ("openai", "openai/gpt-4.1-nano"),
]
PROMPTS = [
    "Explain in two sentences what a balance sheet is.",
    "Write a short professional email asking a professor for an extension on a report.",
    "A company has 120 employees and 35% work remotely. How many work on site?",
]
OUTPUT_FILE = Path(__file__).parent / "comparing_llms.csv"
HEADER = ["model", "prompt", "answer", "input_tokens", "output_tokens", "cost_usd", "latency_seconds"]

load_dotenv()  # reads the .env file


def main():
    rows = []
    for provider, model in MODELS:
        # no cache: we want to measure the real cost and latency of each model
        llm = LLMWrapper(provider, model=model, use_cache=False)
        for prompt in PROMPTS:
            response = llm.ask(prompt)
            logger.info("{} | {!r}: {}", model, prompt, response)
            rows.append([
                model,
                prompt,
                response.text,
                response.input_tokens,
                response.output_tokens,
                response.cost_usd,
                round(response.latency_seconds, 3),
            ])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(HEADER)
        writer.writerows(rows)
    logger.info("Saved {} rows to {}", len(rows), OUTPUT_FILE)

    costs, latencies = defaultdict(list), defaultdict(list)
    for model, _, _, _, _, cost, latency in rows:
        costs[model].append(cost)
        latencies[model].append(latency)
    for model in costs:
        logger.info(
            "{:<30} average cost ${:.6f}  average latency {:.2f} s",
            model,
            sum(costs[model]) / len(costs[model]),
            sum(latencies[model]) / len(latencies[model]),
        )


if __name__ == "__main__":
    main()

# Results (2026-10-14, one run, 3 prompts per model):
#   model                        avg cost     avg latency
#   anthropic/claude-haiku-4.5   $0.000586    2.07 s
#   anthropic/claude-sonnet-5.5  $0.001836    2.28 s
#   openai/gpt-4.1-mini          $0.000164    1.90 s
#   openai/gpt-4.1-nano          $0.000040    2.38 s  (one 5.0 s outlier, ~1.05 s otherwise)
#
# - Cheapest: gpt-4.1-nano, about 45x cheaper than Sonnet. Fastest: gpt-4.1-nano on most calls,
#   gpt-4.1-mini on average. With 3 calls per model, latency is noisy: one slow call changes the ranking.
# - The cheapest answer is good enough for all 3 prompts: correct definition, usable email, 78 on site.
#   The prompts are easy; the differences show in the details (Sonnet's email is the most complete,
#   nano writes LaTeX like \boxed{78} that a plain-text app would have to clean).
# - Real-time customer answers: a fast and cheap model (gpt-4.1-mini or Haiku) with streaming,
#   so the customer sees the first words at once. Moving to a bigger model only if quality is lacking.
# - 10,000 documents overnight: latency does not matter, cost does. The cheapest model that passes a
#   quality check on a sample (here nano: 10,000 x $0.00004 = $0.40, vs about $18 with Sonnet),
#   with retries for the rate limit and a cache so a crash does not pay twice.
