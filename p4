from google.colab import files
import os

print("Please upload your 'api_key.txt' file:")
uploaded = files.upload()

try:
    with open("api_key.txt", "r") as file:
        api_key = file.read().strip()
    print("API Key loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find 'api_key.txt'. Please ensure the file is named correctly.")


try:
    import openai
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "honesty is the best"}
        ],
        n=3,              # Generate 3 responses
        temperature=1,    # Level of creativity
        store=True        # Store the conversation if supported
    )

    print("\n--- Results ---\n")
    for i in range(len(completion.choices)):
        print(f"--- Response {i+1} ---")
        print(completion.choices[i].message.content)
        print("\n") # Adds a new line for better separation

except Exception as e:
    print(f"An error occurred: {e}")
