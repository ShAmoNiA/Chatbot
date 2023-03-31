import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import pickle
import random

# load the words list and labels list from files
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

# load the saved model
model = tf.keras.models.load_model('chatbot_model.h5')

# load the responses from file
with open('responses.pkl', 'rb') as file:
    responses = pickle.load(file)

# define a function to clean and tokenize user input
def clean_input(input_text):
    lemmatizer = WordNetLemmatizer()
    # tokenize the input
    tokens = nltk.word_tokenize(input_text)
    # lemmatize each word in the input
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return tokens

# define a function to convert user input into a bag of words vector
def create_bag_of_words(input_text):
    # create an array of 0s with the length of the word list
    bag_of_words = np.zeros(len(words), dtype=np.float32)
    # clean and tokenize the input
    tokens = clean_input(input_text)
    # loop through each word in the input and set the corresponding index to 1 in the bag_of_words array
    for token in tokens:
        for i, word in enumerate(words):
            if word == token:
                bag_of_words[i] = 1
    return bag_of_words

# define a function to predict the label of user input
def predict_label(input_text):
    # convert the input to a bag of words vector
    bag_of_words = create_bag_of_words(input_text)
    # predict the label using the saved model
    predictions = model.predict(np.array([bag_of_words]))
    # get the index of the highest probability prediction
    prediction_index = np.argmax(predictions)
    # return the corresponding label
    return labels[prediction_index]

# define a function to get a random response for a given label
def get_response(label):
    if label not in responses:
        return "I'm sorry, I don't know how to respond to that."
    response = random.choice(responses[label])
    return response

# define a function to handle user input and generate a response
def chat():
    print("Start talking with the bot (type 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        label = predict_label(user_input)
        response = get_response(label)
        print(response)

# run the chatbot
chat()
