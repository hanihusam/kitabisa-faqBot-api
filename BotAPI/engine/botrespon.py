# yang dibutuhkan di NLP
# Stemmer library dari Lancaster
import json
import os
import pickle
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()

# Stemmer library dari Sastrawi (using Algoritma Nazief Adriani)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# yang dibutuhkan di Tensorflow

# ngambil semua training data yang udah terstruktur
modulePath = os.path.dirname(__file__)  # get current directory
filePath = os.path.join(modulePath, 'training_data')
data = pickle.load(open(filePath, "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import chatbot intents file
dataPath = os.path.join(modulePath, 'intents.json')
with open(dataPath, 'r+', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# build NN nya
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# deskripsi model dan setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    # tokenize polanya
    sentence_words = nltk.word_tokenize(sentence)
    # stem setiap kata
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# mengembalikan kumpulan kata2 array: 0 atau 1 untuk setiap kata pada kalimat


def bow(sentence, words, show_details=False):
    # tokenize polanya
    sentence_words = clean_up_sentence(sentence)
    # kumpulan kata
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


# load model yang tersimpan
model.load(os.path.join(modulePath, './model.tflearn'))
# membuat struktur data untuk mengatasi konteks dari user
context = {}

ERROR_THRESHOLD = 0.5


def classify(sentence):
    # generate probabilitas dari model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # apabila kita memiliki klasifikasinya maka memadankan dengan intent tag
    if results:
        # loop sampai ada yang cocok
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context untuk intent ini jika diperlukan
                    if 'context_set' in i:
                        if show_details:
                            print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check apabila intent nya masuk ke dalam konteks and apply ke user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print('tag:', i['tag'])
                        # random respon dari intent
                        return random.choice(i['responses'])

            results.pop(0)
