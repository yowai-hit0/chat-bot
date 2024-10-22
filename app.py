from flask import Flask, request, jsonify
import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import random

# Load the model and other necessary files
model = load_model('./model/chatbot_model.keras')

# Load intents
with open('./model/intents.json') as file:
    intents = json.load(file)

# Load words and classes
with open('./model/words.pkl', 'rb') as fl:
    words = pickle.load(fl)

with open('./model/classes.pkl', 'rb') as fl:
    classes = pickle.load(fl)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_input(user_input):
    word_list = nltk.word_tokenize(user_input)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]

    bag = [0] * len(words)
    for w in word_list:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(user_input):
    bag = preprocess_input(user_input)
    res = model.predict(np.array([bag]))[0]
    return classes[np.argmax(res)], np.max(res)


def get_response(predicted_class):
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            return np.random.choice(intent['responses'])


def get_recommendations():
    all_patterns = [pattern for intent in intents['intents'] for pattern in intent['patterns']]
    return random.sample(all_patterns, random.randint(1, 3))


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    predicted_class, confidence = predict_class(user_input)
    if confidence > 0.5:  # Confidence threshold
        response = get_response(predicted_class)
    else:
        response = "I didn't understand that. Can you rephrase?"

    recommendations = get_recommendations()  # Get 3 recommendations
    return jsonify({
        'response': response,
        'recommendations': recommendations
    })


@app.route('/chat', methods=['GET'])
def recommendations():
    recommendations = get_recommendations()
    return jsonify({'recommendations': recommendations})


if __name__ == "__main__":
    app.run(debug=True)
