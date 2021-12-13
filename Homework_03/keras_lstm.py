import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import random
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import multiprocessing

import torch.nn as nn
import torch
from torchvision import transforms,datasets
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from keras.initializers import constant
import sys
sys.setrecursionlimit(1000000)
import keras
import yaml

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
gpu = gpus[0]
tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

TRAIN_RATE = 0.8
TEST_RATE = 0.2

# 文件路径
TRAIN_DIR = "./data/train"
TRAIN_POS_DIR = "./data/train/1"
TRAIN_NEG_DIR = "./data/train/0"
VAL_DIR = "./data/val"
VAL_POS_DIR = "./data/val/1"
VAL_NEG_DIR = "./data/val/0"
TEST_DIR = "./data/test"
TEST_POS_DIR = "./data/test/1"
TEST_NEG_DIR = "./data/test/0"

cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 200
n_iterations = 1  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
n_epoch = 5
input_length = 200
maxlen = 200
batch_size = 32

class DataPreprocess():
    def __init__(self, path, vocab_dim = 100, n_iteration = 1, n_exposure = 10,
                 window_size = 7, n_epoch = 4, input_len = 100, max_len = 100):
        '''
        initialize params
        path: csv path
        vocab_dim: dim of vocabulary
        n_iteration:
        n_exposure: the times that words come up
        window_size:
        input_length: length of input
        max_len: max len of vocabulary
        '''
        self.csv_path = path
        self.vocab_dim = vocab_dim
        self.n_iteration = n_iteration
        self.n_exposure = n_exposure
        self.window_size = window_size
        self.n_epoch = n_epoch
        self.input_len = input_len
        self.max_len = max_len

    def loadData(self):
        '''
        load csv file
        :return:
        '''
        contents = pd.read_csv(self.csv_path + "weibo_senti_100k.csv")
        return contents

    def divideData(self, sentences):
        sentences_num = sentences.shape[0]
        sentence_label = []
        for i in range(sentences_num):
            sentence_label.append([sentences['review'][i], sentences['label'][i]])
        random.shuffle(sentence_label)
        train_sentence_label = sentence_label[0:int(TRAIN_RATE * sentences_num)]
        test_sentence_label = sentence_label[int(TRAIN_RATE * sentences_num):sentences_num]
        train_data, train_label= self.cutSentences(train_sentence_label)
        test_data, test_label = self.cutSentences(test_sentence_label)
        return train_data, train_label, test_data, test_label

    def cutSentences(self, text):
        '''
        cut sentences by jieba
        :param text:  list of text
        :return:
        '''
        newText = [jieba.lcut(line.replace("\n", "")) for line, _ in text]
        labels = [label for _, label in text]
        return newText, labels

    def parseDatasets(self, sentences, w2index):
        data = []
        for sentence in sentences:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2index(word))
                except:
                    new_txt.append(0)
            data.append(new_txt)
        return data

    def buildDictionary(self, model, sentences):
        '''
        build words dictionary
        :param model: word2vec model
        :param sentences: list of text
        :return:
        '''
        gensim_dic = Dictionary()
        # 划分词袋
        # print(model.wv.index_to_key)
        gensim_dic.doc2bow(model.wv.index_to_key, allow_update=True)
        # 寻找词频数超过10的索引和向量
        # w2index: 单词以及出现的次数
        # w2vec: 单词对应向量
        w2index = {v: k + 1 for k, v in gensim_dic.items()}
        w2vec = {word: model.wv[word] for word in w2index.keys()}
        parse_sent = self.parseDatasets(sentences, w2index)
        # 把序列填充成长度相同
        parse_sent = sequence.pad_sequences(parse_sent, maxlen=maxlen)
        return w2index, w2vec, parse_sent

    def buildWordVector(self, sentences):
        '''
        convert words to vectors by Word2Vec
        :param sentences: list of txt
        :return:
        '''
        model = Word2Vec(vector_size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         epochs=n_iterations)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save('./lstm_data/word2vec.pkl')
        dic_indexes, word_vectors, parse_sent = self.buildDictionary(model=model, sentences=sentences)
        return dic_indexes, word_vectors, parse_sent

    def getData(self, dic_index, word_vectors, sentences, y):
        # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        n_symbols = len(dic_index) + 1
        # 索引为0的词语，词向量全为0
        embedding_weights = np.zeros((n_symbols, vocab_dim))
        # 从索引为1的词语开始，对每个词语对应其词向量
        for word, index in dic_index.items():
            embedding_weights[index, :] = word_vectors[word]
        # 训练验证3:1
        x_train, x_val, y_train, y_val = train_test_split(sentences, y, test_size=0.25)
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        y_test = keras.utils.to_categorical(y_val, num_classes=2)
        # print x_train.shape,y_train.shape
        return n_symbols, embedding_weights, x_train, y_train, x_val, y_val
class MyTorchModule(nn.Module):
    def __init__(self):
        super(MyTorchModule, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_symbols, embedding_dim=vocab_dim)
        self.lstm = nn.Sequential(nn.LSTM(vocab_dim, 3, 20), nn.Tanh(), nn.Dropout(0.5))
        self.fc = nn.Sequential(nn.Linear(2), nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        output = self.fc(x)
        return output

def lstm_train(n_symbols, weights, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(input_dim=n_symbols,
                        output_dim=vocab_dim,
                        # mask_zero=True,
                        embeddings_initializer=constant(weights),
                        input_length=input_length))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Training:")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)
    print("Test:")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('./model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('./model/lstm.h5')
    print("Test score: ", score)

if __name__ == '__main__':
    # divideDatasetsFromCSV(r"H:\个人资料\研究生\研究生课件\深度学习\作业\第三次\weibo_senti_100k\\")
    demo = DataPreprocess(r"H:\个人资料\研究生\研究生课件\深度学习\作业\第三次\weibo_senti_100k\\")
    print("Loading data:")
    # sentences, labels = loadData(r"H:\个人资料\研究生\研究生课件\深度学习\作业\第三次\weibo_senti_100k\\")
    sentences = demo.loadData()
    train_val_data, train_val_label, test_data, test_label = demo.divideData(sentences)
    index_dict, word_vectors, sentences = demo.buildWordVector(train_val_data)
    n_symbols, embedding_weights, x_train, y_train, x_val, y_val = \
        demo.getData(index_dict, word_vectors, sentences, train_val_label)
    lstm_train(n_symbols, embedding_weights, x_train, y_train, x_val, y_val)