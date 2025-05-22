import spacy
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict
import time
from collections import Counter

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

PRODUCT_CATEGORIES = [
    "Pain Relief", "Health & Wellness", "Medications", "Supplements", "Vitamins",
    "Air Purifiers", "Ergonomic Furniture", "Office Chairs", "Smartphones", "Laptops",
    "Beauty Products", "Sleep Aids", "Fitness Equipment", "Mental Health Support",
    "Cleaning Devices", "Dietary Supplements", "Skin Care", "Hair Care",
    "Home Appliances", "Blue Light Glasses", "Back Support", "Eye Care", "Massage Devices",
    "Over-the-counter Medicine", "Cognitive Boosters"
]
CATEGORY_EMBEDDINGS = embedder.encode(PRODUCT_CATEGORIES, convert_to_tensor=True)

from difflib import SequenceMatcher

def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold

def extract_primary_topics(text: str, top_k: int = 3) -> List[str]:
    """
    Extracting primary topics from the conversation using spacy NLP model - nouns, named entity etc and then comparing it with the llm embeddings here I used MiniLM 
    to keep only those keywords which have some contextual similarity is useful for commercial purpose
    """
    doc = nlp(text)
    candidates = set()
    named_entities = set()

    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG", "GPE", "NORP", "PERSON", "FAC", "LAW"} or ent.label_.endswith("LOC"):
            cleaned = ent.text.lower().strip()
            if 2 <= len(cleaned) <= 50 and not cleaned.isdigit():
                named_entities.add(cleaned)
                candidates.add(cleaned)

    for chunk in doc.noun_chunks:
        tokens = [token for token in chunk 
                  if token.pos_ in {"NOUN", "PROPN", "ADJ"} 
                  and not token.is_stop and token.is_alpha]
        if not tokens:
            continue
        lemmatized = " ".join(token.lemma_.lower() for token in tokens)
        if 2 <= len(lemmatized) <= 50:
            candidates.add(lemmatized.strip())

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and token.is_alpha:
            candidates.add(token.lemma_.lower().strip())

    candidates = list(set(c for c in candidates if len(c) > 2 and not c.isdigit()))
    if not candidates:
        return []

    with torch.no_grad():
        embeddings = embedder.encode([text] + candidates, convert_to_tensor=True)
        similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    ranked = sorted(
        zip(candidates, similarities.tolist()),
        key=lambda x: (x[0] in named_entities, x[1]),
        reverse=True
    )

    final_topics = []
    for phrase, _ in ranked:
        if any(phrase in t or t in phrase or is_similar(phrase, t) for t in final_topics):
            continue
        final_topics.append(phrase)
        if len(final_topics) >= top_k:
            break

    return final_topics


def estimate_commercial_intent(text: str, primary_topics: List[str]) -> float:
    """
    This is a simple commercial intent prediction which just checking keywords in it to score
    """
    score = 0.1
    lower_text = text.lower()
    indicators = ["need", "buy", "purchase", "recommend", "looking for", "problem", "tried", "help"]
    for word in indicators:
        if word in lower_text:
            score += 0.1
    if any(t in lower_text for t in primary_topics):
        score += 0.2
    return min(score, 1.0)

def match_product_categories(primary_topics: List[str], threshold: float = 0.5, top_k: int = 5) -> List[str]:
    """
    Matching the primary topics with the product categories that we stored in list using sentence transformer only and comparing it using cosine similarity
    """
    if not primary_topics:
        return []

    normalized_topics = [t.lower().strip() for t in primary_topics]

    with torch.no_grad():
        topic_embeddings = embedder.encode(normalized_topics, convert_to_tensor=True)
        avg_embedding = torch.mean(topic_embeddings, dim=0, keepdim=True)
        similarities = util.cos_sim(avg_embedding, CATEGORY_EMBEDDINGS)[0]

    top_indices = torch.topk(similarities, k=top_k).indices.tolist()
    best_matches = [PRODUCT_CATEGORIES[i] for i in top_indices if similarities[i] >= threshold]
    return best_matches

def analyze_conversation(text: str) -> Dict:
    start_time = time.perf_counter()
    primary_topics = extract_primary_topics(text)
    commercial_intent = estimate_commercial_intent(text, primary_topics)
    product_categories = match_product_categories(primary_topics)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "primary_topics": primary_topics,
        "commercial_intent": round(commercial_intent, 2),
        "product_categories": product_categories,
        "latency_ms": round(latency_ms, 2)
    }

# Test
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

