from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

while True:
    text = input("\nEnter text (or type exit): ")
    if text.strip().lower() == "exit":
        print("Program ended")
        break
    if text.strip() == "":
        print("Please enter some text")
        continue
    result = sentiment_pipeline(text)
    print("Sentiment:", result[0]['label'], "(Confidence:", round(result[0]['score'], 2), ")")
