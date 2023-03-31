## Chatbot

This is a simple chatbot that can communicate with users and provide helpful responses, using natural language processing and machine learning algorithms. The chatbot is built using Python and the TensorFlow library.

## Installation
To use the chatbot, you will need to install Python and TensorFlow. You can install Python from the official website (https://www.python.org/downloads/) and TensorFlow using pip:

```
pip install tensorflow
```
You will also need to download the following files:

1.chatbot.py: the main Python script that runs the chatbot
2.words.pkl: a pickled list of all the words used in the training data
3.labels.pkl: a pickled list of all the labels used in the training data
4.chatbot_model.h5: the saved TensorFlow model used for prediction
5.responses.pkl: a pickled dictionary of responses for each label
## Usage
To start the chatbot, simply run the chatbot.py script from the command line:

```
python chatbot.py
```
The chatbot will start running and prompt you to start talking. Simply type your message and the chatbot will respond with a helpful response.

To exit the chatbot, type "quit" and press enter.

## Customization
If you want to customize the chatbot, you can modify the responses.pkl file to add your own responses. Simply open the file in a text editor and add new responses to the relevant label.

You can also train your own model by modifying the train_chatbot.py script and running it. This will generate new words.pkl, labels.pkl, and chatbot_model.h5 files that you can use with the chatbot.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
