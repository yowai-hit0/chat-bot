import nltk
import pickle
import json
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)


model = load_model('chatbot_model.keras')

def preprocess_input(user_input):
    # Tokenize and lemmatize input
    word_list = nltk.word_tokenize(user_input)
    word_list = [lematizer.lemmatize(word.lower()) for word in word_list]

    # Create a bag of words
    bag = [0] * len(words)
    for w in word_list:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)


def get_response(user_input):
    # Preprocess user input
    input_data = preprocess_input(user_input)
    input_data = input_data.reshape(1, -1)

    # Get model prediction
    prediction = model.predict(input_data)
    intent_index = np.argmax(prediction)

    # Get the corresponding intent
    tag = classes[intent_index]

    # Find the corresponding response
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            return [response, tag]

    return "I'm sorry, I didn't understand that."


if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        response = get_response(user_input)
        if user_input.lower() == 'quit':
            break
        print(f"Chatbot: {response[0]}")
        if response[1] == 'goodbye':
            break

