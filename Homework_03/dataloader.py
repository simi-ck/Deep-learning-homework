import pandas as pd
import os
import jieba
import random
from gensim.models.word2vec import Word2Vec
import torch
import torch.nn as nn
import numpy as np

TRAIN_RATE = 0.6
VAL_RATE = 0.2
TEST_RATE = 0.2

class DataLoader():
    def __init__(self, csv_path, sen_len, w2v_path="./model/word2vec.model"):
        self.csv_path = csv_path
        # 词向量路径
        self.w2v_path = w2v_path
        # 句子长度
        self.sen_len = sen_len
        # 根据列表索引查找单词  word
        self.idx2word = []
        # 单词和索引的字典 word: index
        self.word2idx = {}
        # embedding矩阵  单词数 X embedding_dim维
        self.embedding_matrix = []
        self.sentences = None
    def loadData(self):
        '''
        load csv file
        :return:
        '''
        contents = pd.read_csv(self.csv_path + "weibo_senti_100k.csv")
        return contents

    def loadSentences(self, sentences):
        self.sentences = sentences

    # CSV 表格形式分割数据
    def cutChineseSentences(self, text):
        '''
        cut sentences by jieba
        :param text:  list of text
        :return:
        '''
        words = [jieba.lcut(line.replace("\n", "")) for line, _ in text]
        labels = [label for _, label in text]
        return words, labels

    def cutEnglishSentences(self, text):
        words = [text.strip("\n").split(" ") for line, _ in text]
        labels = [label for _, label in text]
    # 中文划分 用jieba
    def divideChineseData(self, contents):
        '''
        3：1：1划分训练、验证和测试
        返回用结巴处理过的词向量列表和对应标签
        :key
        '''
        review_num = contents.shape[0]
        review_label = []
        for i in range(review_num):
            review_label.append([contents['review'][i], contents['label'][i]])
        # 把数据打乱
        random.shuffle(review_label)
        # 训练验证review和label
        train_val_review_label = review_label[0:int((TRAIN_RATE + VAL_RATE) * review_num)]
        # 测试review和label
        test_review_label = review_label[int((TRAIN_RATE + VAL_RATE)* review_num):review_num]
        # 用jieba划分中文
        train_val_data, train_val_label = self.cutChineseSentences(train_val_review_label)
        test_data, test_label = self.cutChineseSentences(test_review_label)
        # 返回训练、验证和测试标签
        return train_val_data, train_val_label, test_data, test_label

    # txt形式加载数据
    def load_train_data(path='training_label.txt'):
        '''
        按照有无标签读取数据 有标签返回单词列表和标签
        无标签返回单词列表
        '''
        if 'training_label' in path:
            with open(path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
            return x, y
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                x = [line.strip('\n').split(' ') for line in lines]
            return x

    def load_testing_data(path='testing_data'):
        '''
        读取测试数据
        :return:
        '''
        with open(path, 'r') as f:
            lines = f.readlines()
            X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
            X = [sen.split(' ') for sen in X]
        return X

    def train_word2vec(self, x):
        '''
        利用Word2Vec训练单词转换成vector的embedding
        '''
        # 训练word to vector 的 word embedding
        # vector_size 特征向量维度
        # window 窗口大小
        # wordkers 线程数
        # min_count 单词出现少于该次数会被忽略
        # sg = 1表示skip gram 其他情况CBOW
        # skip gram  通过目标单词推测语境
        # CBOW 通过语境推测单词
        model = Word2Vec(vector_size=250, window=5, min_count=5, workers=12, epochs=10, sg=1)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("./model/word2vec.model")

    def get_w2v_model(self):
        '''
        加载之前保存模型
        '''
        # 加载词向量模型
        self.embedding = Word2Vec.load(self.w2v_path)
        # 词向量大小
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        '''
        # 对每个给定word从embedding中加载对应词向量并且加入embedding_matrix
        # word只會是"<PAD>"或"<UNK>"
        # 生成1 X embedding_dim行向量
        '''
        vector = torch.empty(1, self.embedding_dim)
        # 给vector正太初始化向量
        torch.nn.init.uniform_(vector)
        # 每添加一个词对应index为字典长度 word1: 1, word2:2 这种形式
        self.word2idx[word] = len(self.word2idx)
        # 把word加入idx2word word1在列表中索引为1 方便和word2idx对照
        self.idx2word.append(word)
        # 在列方向上拼接 0 列方向拼接 1 行方向上拼接
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        '''
        制作embedding vector
        '''
        print("Get embedding ...")
        self.embedding_matrix = []
        # 加载模型
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作 word2idx 的 dictionary
        # 制作 idx2word 的 list
        # 制作 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.index_to_key):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            # 对每个单词添加单词对应词向量
            self.embedding_matrix.append(self.embedding.wv[word])
        # 把embedding_matrix转换成torch.tensor形式
        self.embedding_matrix = torch.tensor(np.array(self.embedding_matrix))
        # 將"<PAD>"跟"<UNK>"加進embedding裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # embedding层要求句子长度一样 需要把所有句子长度填充长度相同
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        # 判断是否达到所需长度
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子里单词转换成索引
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 填充句子长度至相同
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把labels转换成tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)