from transformers import pipeline

# Create a sentiment analysis pipeline
# The 'distilbert-base-uncased-finetuned-sst-2-english' model is great for this task.
sentiment_analyzer = pipeline("sentiment-analysis")

print("Welcome to the Sentiment Analyzer! Type 'exit' to quit.")
print("-" * 50)

while True:
    # Get user input
    user_input = input("Enter a sentence to analyze: ")

    # Check for exit command
    if user_input.lower() == 'exit':
        print("Exiting program.")
        break

    # Run the analysis on the user's sentence
    results = sentiment_analyzer(user_input)

    # Print the result
    result = results[0]  # The pipeline returns a list of dictionaries, we just need the first one
    sentiment = result['label']
    score = result['score']

    print(f"\nSentiment: {sentiment}")
    print(f"Score: {score:.4f}\n")
    print("-" * 50)