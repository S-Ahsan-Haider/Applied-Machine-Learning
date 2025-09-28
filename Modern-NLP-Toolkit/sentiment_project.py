from textblob import TextBlob

# A loop that continues until the user exits
while True:
    # Get text input from the user
    user_input = input("Enter a sentence to analyze (or type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        print("Exiting program.")
        break
    
    # Check if the input is not empty
    if user_input.strip() == "":
        print("Input cannot be empty. Please try again.")
        continue

    # Create a TextBlob object from the user's input
    blob = TextBlob(user_input)
    
    # Get the polarity score
    polarity = blob.sentiment.polarity
    
    # Determine the sentiment
    sentiment = "neutral"
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
        
    # Print the results
    print(f"\nYour sentence: '{user_input}'")
    print(f"Sentiment: {sentiment.upper()}")
    print(f"Polarity score: {polarity}\n")