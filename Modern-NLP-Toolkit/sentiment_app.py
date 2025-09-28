# Import necessary components from the Flask and Transformers libraries.
from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask application. This is the main object that will run the web server.
app = Flask(__name__)

# Load the sentiment analysis model from Hugging Face.
# We do this once when the application starts to save time and resources.
# The model will stay in memory, ready to be used for every request.
sentiment_analyzer = pipeline("sentiment-analysis")

# Define an API endpoint using a Flask decorator.
# This specifies that when a POST request is sent to the "/analyze-sentiment" URL,
# the 'analyze_sentiment' function below will be executed.
@app.route("/analyze-sentiment", methods=["POST"])
def analyze_sentiment():
    """
    Handles POST requests to analyze the sentiment of a sentence.
    It expects a JSON body with a 'sentence' key.
    """
    # Try to get the JSON data from the request body.
    try:
        data = request.json
        # Get the value associated with the 'sentence' key.
        sentence = data.get("sentence")
    except Exception as e:
        # If the request body is not valid JSON, return a 400 Bad Request error.
        return jsonify({"error": f"Invalid JSON provided: {e}"}), 400

    # Check if a sentence was actually provided in the JSON data.
    if not sentence:
        # If the 'sentence' key is missing or empty, return a 400 Bad Request error.
        return jsonify({"error": "No sentence provided in the request body."}), 400

    # Run the analysis on the sentence using the pre-loaded model.
    results = sentiment_analyzer(sentence)

    # The pipeline returns a list of one dictionary. We grab that dictionary.
    result = results[0]

    # Prepare the response as a Python dictionary.
    response_data = {
        "input_sentence": sentence,
        "sentiment": result["label"],
        "score": result["score"]
    }

    # Use jsonify to convert the dictionary into a proper JSON response and send it back.
    return jsonify(response_data)

# This block ensures that the app only runs when this script is executed directly,
# not when it's imported as a module.
if __name__ == "__main__":
    # Start the Flask development server.
    # host='0.0.0.0' makes the server accessible from any device on your local network.
    # port=5000 sets the port number for the server.
    app.run(host='0.0.0.0', port=5000, debug=True)
