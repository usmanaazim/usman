!pip install gensim openai

import gensim.downloader as api
from openai import OpenAI

print("Loading Word2Vec model...")
model = api.load("glove-wiki-gigaword-100")   # Pretrained embeddings
print("Model loaded successfully!\n")

original_prompt = "Explain machine learning in healthcare."

print("Original Prompt:")
print(original_prompt)

keyword = "healthcare"

print("\nRetrieving similar words using Word Embeddings...\n")
similar_words = model.most_similar(keyword, topn=5)

similar_word_list = [word for word, score in similar_words]

print("Similar Words Found:")
for word, score in similar_words:
    print(f"{word}  (Similarity: {score:.4f})")

enriched_prompt = f"""
Explain machine learning in healthcare including aspects such as 
{', '.join(similar_word_list)}. 
Provide detailed examples and applications.
"""

print("\nEnriched Prompt:")
print(enriched_prompt)

client = OpenAI(api_key="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")

def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

print("\nGenerating response for Original Prompt...\n")
original_response = generate_response(original_prompt)

print("Generating response for Enriched Prompt...\n")
enriched_response = generate_response(enriched_prompt)

print("\n==============================")
print("RESPONSE FOR ORIGINAL PROMPT")
print("==============================\n")
print(original_response)

print("\n==============================")
print("RESPONSE FOR ENRICHED PROMPT")
print("==============================\n")
print(enriched_response)


print("\n==============================")
print("COMPARISON")
print("==============================\n")

print("Observation:")
print("1. The enriched prompt typically produces more detailed explanations.")
print("2. It includes related domains from embedding expansion.")
print("3. The content is more structured and application-focused.")
