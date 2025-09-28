import spacy

# Load the small English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Download the 'en_core_web_sm' model first:")
    print("python -m spacy download en_core_web_sm")
    exit()

# A more complex sentence with entities
text = "Apple is looking at buying U.K. startup for $1 billion in London, U.K."

# Process the text with spaCy
doc = nlp(text)

# Print the text and its named entities
print(f"Original Text: {text}\n")
print("Named Entities:")

# Iterate over the entities in the processed document
for ent in doc.ents:
    print(f"  - Entity: '{ent.text}' | Type: '{ent.label_}'")