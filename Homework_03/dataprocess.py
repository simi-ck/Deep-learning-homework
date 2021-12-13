from gensim.models.word2vec import Word2Vec
import torch
import torch.nn as nn
import numpy as np

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./model/word2vec.model"):
        # 词向量路径
        self.w2v_path = w2v_path
        # 句子
        self.sentences = sentences
        # 句子长度
        self.sen_len = sen_len
        # 根据列表索引查找单词  word
        self.idx2word = []
        # 单词和索引的字典 word: index
        self.word2idx = {}
        # embedding矩阵  单词数 X embedding_dim维
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 加载词向量模型
        self.embedding = Word2Vec.load(self.w2v_path)
        # 词向量大小
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # 对每个给定word从embedding中加载对应词向量并且加入embedding_matrix
        # word只會是"<PAD>"或"<UNK>"
        # 生成1 X embedding_dim行向量
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
        print("Get embedding ...")
        # 加载模型
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作 word2idx 的 dictionary
        # 制作 idx2word 的 list
        # 制作 word2vector 的 list
        print(self.embedding.wv.index_to_key)
        for i, word in enumerate(self.embedding.wv.index_to_key):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
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