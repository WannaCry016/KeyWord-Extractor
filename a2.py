import re
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import torch
import requests
import os
import json
from difflib import SequenceMatcher

# Load NLP and embedding models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold

def extract_primary_topics(text, top_k=3):
    """
    Extracts and ranks the most relevant noun-based phrases from the conversation text,
    using semantic similarity and duplicate filtering.
    """
    doc = nlp(text)
    candidates = set()
    named_entities = set()

    # Collect named entities for scoring boost
    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG", "GPE", "NORP", "PERSON", "FAC", "LAW"} or ent.label_.endswith("LOC"):
            cleaned = ent.text.lower().strip()
            if 2 <= len(cleaned) <= 50 and not cleaned.isdigit():
                named_entities.add(cleaned)
                candidates.add(cleaned)

    # Noun phrase extraction
    for chunk in doc.noun_chunks:
        tokens = [token for token in chunk 
                  if token.pos_ in {"NOUN", "PROPN", "ADJ"} 
                  and not token.is_stop and token.is_alpha]
        if not tokens:
            continue
        lemmatized = " ".join(token.lemma_.lower() for token in tokens)
        if 2 <= len(lemmatized) <= 50:
            candidates.add(lemmatized.strip())

    # Fallback: individual nouns and proper nouns
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and token.is_alpha:
            candidates.add(token.lemma_.lower().strip())

    # Filter out short or junk tokens
    candidates = list(set(c for c in candidates if len(c) > 2 and not c.isdigit()))
    if not candidates:
        return []

    # Embed and rank candidates
    with torch.no_grad():
        embeddings = embedder.encode([text] + candidates, convert_to_tensor=True)
        similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    ranked = sorted(
        zip(candidates, similarities.tolist()),
        key=lambda x: (x[0] in named_entities, x[1]),
        reverse=True
    )

    # Remove near-duplicates
    final_topics = []
    for phrase, _ in ranked:
        if any(phrase in t or t in phrase or is_similar(phrase, t) for t in final_topics):
            continue
        final_topics.append(phrase)
        if len(final_topics) >= top_k:
            break

    return final_topics

def rag_method(primary_topics):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"  

    if not api_key:
        raise Exception("GROQ_API_KEY environment variable not set.")

    prompt = f"""
Given the following extracted primary topics from a user conversation, estimate how likely it is that the user has an actual commercial need (buying or researching to buy something) rather than just mentioning the topic casually. and relevant product categories.

Primary Topics: {primary_topics}

Return:
- "commercial_intent" as a float between 0.0 and 1.0, where 1.0 is a strong buying signal.
- "product_categories" as a list of standardized commercial product categories the topic relates to.

Only output valid JSON:
{{
  "commercial_intent": <float>,
  "product_categories": ["..."]
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 150
    }

    start = time.perf_counter()
    response = requests.post(api_url, json=payload, headers=headers)
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise Exception(f"Groq API Error: {response.status_code} - {response.text}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response content.")
        json_str = match.group()
        response_json = json.loads(json_str)
        return response_json, round(latency_ms, 2)
    except Exception as e:
        raise Exception(f"Failed to parse response: {e}\nRaw: {response.text}")

# ---- 3. Main Pipeline ----
def analyze_conversation(conversation_text):
    primary_topics = extract_primary_topics(conversation_text)
    llm_response, latency = rag_method(primary_topics)

    return {
        "primary_topics": primary_topics,
        "commercial_intent": llm_response.get("commercial_intent", 0.0),
        "product_categories": llm_response.get("product_categories", []),
        "latency_ms": latency
    }

# ---- 4. Run Tests ----
if __name__ == "__main__":
    test_cases = [
        "User: I've been having these headaches that won't go away.\nAssistant: I'm sorry to hear that. How long have you been experiencing them?\nUser: About two weeks now. I've tried basic painkillers but nothing helps.",
        "User: Do you know if there's any natural way to boost energy without coffee?\nAssistant: Some people try ginseng or B12. Would you prefer supplements?",
        "User: I just moved to a new apartment and I need furniture.\nAssistant: What kind of furniture are you looking for?\nUser: A sofa, a desk, and maybe a small dining table.",
        "User: I was just talking to my friend about how much we miss playing board games.\nAssistant: Do you still have your old games?\nUser: Yeah, but we haven’t touched them in months.",
        "User: What’s the difference between espresso and drip coffee?\nAssistant: It’s mostly in the preparation method and flavor.\nUser: Ah okay, just curious.",
        "User: I'm looking for a good air purifier for my bedroom.\nAssistant: Do you need one with HEPA filters or smart features?\nUser: HEPA for sure. My allergies are really bad.",
        "User: Do you know any websites that sell ergonomic office chairs?\nAssistant: Yes, a few good ones! Do you prefer mesh or leather?\nUser: Mesh, something breathable.",
        "User: I've been reading about hair thinning in men and treatments.\nAssistant: There are both over-the-counter and prescription options.\nUser: I might talk to a dermatologist first.",
        "User: This new iPhone has some amazing camera features.\nAssistant: Yeah, especially the low-light performance.\nUser: I might wait till next year to upgrade though."
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(analyze_conversation(case))
