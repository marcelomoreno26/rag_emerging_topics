import os
import random
import requests
from datasets import load_dataset

BASE_URL = "http://localhost:8000"

def preprocess_column(texts: list[str]) -> list[str]:
    """
    Remove duplicates, empty entries, and strip whitespace.
    """
    return list(set(text.strip() for text in texts if text and text.strip()))

def upload_texts_to_api(texts: list[str]):
    """
    Upload preprocessed texts to the /upload endpoint.
    The API will handle chunking.
    """
    payload = {"texts": texts}
    response = requests.post(f"{BASE_URL}/upload", json=payload)
    if response.ok:
        print("✅ Upload successful:", response.json())
    else:
        raise Exception(f"❌ Upload failed: {response.status_code}, {response.text}")

def generate_response(question: str):
    """
    Send a single question to the /generate endpoint and print the result.
    """
    payload = {"new_message": {"role": "user", "content": question}}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    if response.ok:
        result = response.json()
        print("\n🧠 Generated Answer:\n", result.get("generated_text"))
        print("\nRetrieved Contexts:\n", result.get("contexts"))
    else:
        raise Exception(f"❌ Generation failed: {response.status_code}, {response.text}")

def main():
    dataset = load_dataset("IIC/RagQuAS", split="test")

    # Step 1: Loop through context_1 to context_5 and upload each column
    for i in range(1, 6):
        column = f"context_{i}"
        if column in dataset.column_names:
            texts = preprocess_column(dataset[column])
            print(f"\n📤 Uploading {len(texts)} texts from column '{column}'...")
            upload_texts_to_api(texts)
        else:
            print(f"⚠️ Column '{column}' not found in dataset. Skipping.")

    # Step 2: Pick one random question and generate response
    question = random.choice(dataset["question"])
    print(f"\n🎯 Asking question: {question}")
    generate_response(question)

if __name__ == "__main__":
    main()
