import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from PIL import Image
import random

# Initialize NLTK
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# <========================================================= Load Resources ===============================================================>
# Load chatbot model and necessary files
model = load_model('chatbot_model.h5')
with open('intents3.json', 'r') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# <---------------------------------------------------------- Page Configuration ----------------------------------------------------------->
im = Image.open('bot.jpg')
st.set_page_config(layout="wide", page_title="Student's Career Counselling Chatbot", page_icon=im)

# <--------------------------------------------------- Hide the Right Side Streamlit Menu Button ------------------------------------------------>
st.markdown(""" <style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style> """, unsafe_allow_html=True)

# <=================================================== Sidebar ==========================================================================>
with st.sidebar:
    st.title('''🤗💬 Student's Career Counselling Bot''')
    st.markdown('''
    ## About~
    This app has been developed by 5 students of VIT-AP :
    - Harshita Bajaj [22MSD7013]
    - Arya Chakraborty [22MSD7020]
    - Rituparno Das [22MSD2027]
    - Shritama Sengupta [22MSD7032]
    - Arundhuti Chakraborty [22MSD7046]
    ''')
    add_vertical_space(5)

# <=================================================== Initializing Session State =========================================================>
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm an AI Career Counselor, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# <==================================================== Input Box Styling =============================================================>
input_container = st.container()
response_container = st.container()

# <================================================= Function to Get Text Input ==========================================================>
def get_text():
    input_text = st.text_input("You: ",  key="input", on_change=None)
    return input_text

# <==================================================== Style Customization for Text Input ===============================================>
styl = f"""
<style>
    .stTextInput {{
    position: fixed;
    bottom: 20px;
    z-index: 20;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# <===================================================== Displaying Input and Response ===================================================>
with input_container:
    user_input = get_text()

# <===================================================== Bot Response Generation ==============================================>
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
    print(f"Prediction output: {res}")  # Debugging output
    ERROR_THRESHOLD = 0.25  # You can adjust this threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    """Return a random response for the predicted intent."""
    tag = ints[0]['intent']
    print(f"Predicted intent: {tag}")  # Log the predicted tag
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print(f"Response chosen: {response}")  # Log the response
            return response
    return "Sorry, I didn't understand that."

def generate_response(user_input):
    """Generate a response from the chatbot."""
    ints = predict_class(user_input, model)  # Get prediction
    response = get_response(ints, intents)  # Get response based on predicted class
    return response

# <==================================================== Submit Button Styling =================================================>
submit_button = st.button("Enter")
styl = f"""
    <style>
        .stButton {{
        position: fixed;
        font-weight: bold;
        margin-top: -10px;
        bottom: 20px;
        left: 1213px;
        font-size: 24px;
        z-index: 9999;
        border-radius: 20px;
        height:200px;
        width:100px;
        }}
    </style>
"""
st.markdown(styl, unsafe_allow_html=True)

# <=================================================== Conditional Display of AI Response ==============================================>
with response_container:
    if user_input: 
        if submit_button:
            if user_input == "Who is your maker":
                response = "GOD!!"
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
            else:
                response = generate_response(user_input)  # Call the function to generate response
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))
