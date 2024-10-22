import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# downlad nltk dependency
nltk.download('punkt')
nltk.download('wordnet')

# variables to use
intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?', '.', ',', '!']
lemmatizer = WordNetLemmatizer()

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# training data
training = []
output_empty = [0] * len(classes)


for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer_sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=50, verbose=1)
model.save('chatbot_model.keras')
print('Model saved! 😄😄😄')

