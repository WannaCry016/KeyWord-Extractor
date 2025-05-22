import re
import time
import requests
import json

def rag_method(conversation_text):
    """
    Sends the full conversation to the LLM through GROQ API and gets an output as a json 
    """

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"  

    prompt = f"""
Given the following user-assistant conversation, perform the following tasks:

1. Extract up to 3 relevant primary topics or items the user seems focused on (e.g., products, services, needs).
2. Estimate the commercial intent as a float between 0.0 and 1.0 — where 1.0 means the user is actively seeking to purchase or research something to buy.
3. Suggest relevant standardized product categories.

Conversation:
\"\"\"
{conversation_text}
\"\"\"

Only return strictly in valid JSON in the format:
{{
  "primary_topics": ["..."],
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
        "max_tokens": 200
    }

    start = time.perf_counter()
    response = requests.post(api_url, headers=headers, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise Exception(f"Groq API Error: {response.status_code} - {response.text}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response content.")
        parsed = json.loads(match.group())
        parsed["latency_ms"] = round(latency_ms, 2)
        return parsed
    except Exception as e:
        raise Exception(f"Failed to parse response: {e}\nRaw: {response.text}")

# ---- Test Cases ----
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
        print(rag_method(case))
