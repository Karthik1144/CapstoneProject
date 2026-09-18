import time
import ollama

MODEL = "llama3.1:8b"

print("Testing Ollama...")
print("Available models:")

models = ollama.list()
for model in models.models:
    print(" -", model.model)

print("\nStarting generation...")

start = time.time()

try:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly one word: hello"
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 16,
        },
    )

    elapsed = time.time() - start

    print("\nSUCCESS")
    print("Response:", response["message"]["content"])
    print(f"Time: {elapsed:.2f} seconds")

except Exception as e:
    elapsed = time.time() - start
    print("\nFAILED")
    print("Error:", repr(e))
    print(f"Time before failure: {elapsed:.2f} seconds")