# Importing packages and modules

import nltk
#nltk.download("punkt")
from nltk import sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow
import random
import json
from tensorflow.python.framework import ops
import pickle

# Json file loading
with open("data.json") as file:
    data = json.load(file)

try:
    with open("pickle_data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
        # List of all words and their labels
    words = []
    labels = []
    docs_x = []  # List of all patterns
    docs_y = []  # Corresponding entries in docs_x

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

        # Creating training and testing lists
    training = []
    output = []
    empty_out = [0 for _ in range(len(labels))]

        # Creating data to feed the model
    for x, doc in enumerate(docs_x):
        new_word = []

        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                new_word.append(1)
            else:
                new_word.append(0)

        output_row = empty_out[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(new_word)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)


    with open("pickle_data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
#tensorflow.reset_default_graph()
ops.reset_default_graph()
# Defining DNN model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation= "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=600, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    new_words = nltk.word_tokenize(s)
    new_words = [stemmer.stem(word.lower()) for word in new_words]

    for sent in new_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("String ChatX")
    print("Type quit to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]

        results_index = np.argmax(results)
        tag = labels[results_index]

        if results(results_index >0.7):
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print("I didn't get that. Can you try another way?")
chat()