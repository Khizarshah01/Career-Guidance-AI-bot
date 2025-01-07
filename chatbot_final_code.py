# <========================================================= Importing Required Libraries =================================================>
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# <========================================================= Initializing NLTK and Lemmatizer =================================================>
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# <========================================================= Loading Intents and Pickle Files =================================================>
# Load intents JSON and pre-trained pickle files for words and classes
with open('intents3.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# <========================================================= Data Preprocessing =================================================>
# Prepare data for training
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word and add to the corpus
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        # Add tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

print(f"Documents: {len(documents)}")
print(f"Classes: {len(classes)} - {classes}")
print(f"Unique Words: {len(words)} - {words}")

# Save processed data into pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# <========================================================= Data Preparation for Training =================================================>
train_x = []
train_y = []

# Iterate over each document and create the training dataset
for doc in documents:
    bag = []
    output_row = [0] * len(classes)

    # Tokenize and lemmatize words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    # Create a bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Set the appropriate output value for the current tag
    output_row[classes.index(doc[1])] = 1

    train_x.append(bag)
    train_y.append(output_row)

# Convert to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")

# <========================================================= Model Creation =================================================>
def create_model(input_shape, output_shape):
    """Create and compile the neural network model."""
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    # Compile the model with SGD optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# <========================================================= Training the Model =================================================>
# Create the model
model = create_model(len(train_x[0]), len(train_y[0]))

# Train the model
model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')
print("Model trained and saved as 'chatbot_model.h5'")

# <========================================================= Chatbot Response Functions =================================================>
def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """Return a bag of words: 0 or 1 for each word in the vocabulary that exists in the sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the class of the input sentence using the trained model."""
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    """Return a random response for the predicted intent."""
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(text):
    """Generate a chatbot response given the user input."""
    ints = predict_class(text, model)
    return get_response(ints, intents)

# Example usage:
# response = chatbot_response("Hello, what courses do you offer?")
# print(response)
