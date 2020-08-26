# yang dibutuhin untuk NLP
# Stemmer library dari Lancaster
import pickle
import json
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

# yang dibutuhin untuk Tensorflow

# import chatbot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)
words = []
classes = []
documents = []
ignore_words = ['?']
# loop setiap kalimat yang ada pada pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize setiap kata di setiap kalimat
        w = nltk.word_tokenize(pattern)
        # masukkan ke words list
        words.extend(w)
        # masukkan ke documents ke dalam corpus
        documents.append((w, intent['tag']))
        # masukkan ke classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem dan lower setiap kata dan hapus kalau ada yang duplikat
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# hapus duplikat
# classes = sorted(list(set(classes)))

# cek hasil tokenize yang sudah diklasifikasikan
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

# create training data
training = []
output = []
# create array kosong untuk output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output nya '0' untuk setiap tag dan '1' untuk current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle features dan balikin ke np.array
random.shuffle(training)
training = np.array(training)

# create train dan test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# simpan semua data structures
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x,
             'train_y': train_y}, open("training_data", "wb"))
