"""
LAB 5 - Your RAG, on your own (about 15 minutes, calls the model)

Goal: use your RAG on questions of your own, and check that the citations really support the answers.

What to do:
  1. Write 3 questions in QUESTIONS. At least one must need both regulations, for example about
     personal data in AI systems. Use your own words, in English.
  2. Run this file. For each question it prints the retrieved chunks, the answer, the citations, and the text of the
     first cited chunk.
  3. Read the first cited chunk of one answer. Does it really support the answer? Write your verdict at the bottom.

Bonus: change k (the number of chunks), or switch PROVIDER in lab_tools.py. What changes?

EXIT TICKET (no notes, answer on paper)
  1. What does retrieval guarantee, and what does it not guarantee?
  2. Why add BM25 to dense search?
  3. Why check the citations in code?
"""
from lab3_hybrid_search import fuse
from lab4_grounded_answer import RULES
from lab_tools import ask_rag, get_corpus, get_llm
from loguru import logger

QUESTIONS = [
    "How long does a company have to report a personal data breach to the authority?",
    "Which AI practices are completely forbidden?",
    "Can a high-risk AI system be trained on sensitive personal data such as health data?",
]

# Everything below only runs when you run this file directly.
if __name__ == "__main__":
    if RULES.startswith("TODO"):
        raise SystemExit("Finish lab 4 first (or copy RULES from the display), then run this file again.")
    if QUESTIONS[0].startswith("TODO"):
        raise SystemExit("TODO not done yet: write your 3 questions")

    corpus = get_corpus()
    llm = get_llm()
    for question in QUESTIONS:
        logger.info(f"===== {question}")
        answer = ask_rag(llm, corpus, question, RULES, fuse, verbose=True)
        logger.info(f"answer: {answer.answer}")
        logger.info(f"citations: {answer.citations}")
        if answer.citations:
            first = corpus.by_id[answer.citations[0]]
            logger.info(f"--- first cited chunk: {first['id']} ---")
            logger.info(first["text"])

# My verdict: does the first cited chunk support the answer? =
# Q1 (data breach): yes, GDPR-Art33-1 says "not later than 72 hours" word for word. Fully supported.
# Q2 (forbidden practices): no. The first cited chunk (AIAct-Art5-8) only says Article 5 does not affect other laws.
#    Retrieval missed the main list of Article 5(1); the model honestly says the sources are incomplete,
#    but still sets answerable=True. Valid citations (in code) do not mean a good answer.
# Q3 (health data, meant to need both laws): AIAct-Art4a-2 supports the "exceptionally, for bias detection" part.
#    But only AI Act chunks were retrieved: GDPR Article 9 (special categories) was missed, so the answer is partial.
#
# EXIT TICKET
# 1. Retrieval guarantees that the model SEES relevant text, with ids we can check. It does not guarantee that the
#    right chunk is found, nor that the model uses it correctly (Q2 above).
# 2. Dense search compares meanings and misses exact numbers, codes and names; BM25 catches them (lab 3, question 2).
# 3. A model can invent an id or cite a chunk it never received; code checks it for free and every time,
#    instead of trusting the model.
