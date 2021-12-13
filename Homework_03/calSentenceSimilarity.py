import gensim
import jieba
# 训练样本
from gensim import corpora
from gensim.similarities import Similarity


# fin = open("questions.txt",encoding='utf8').read().strip(' ')   #strip()取出首位空格
# jieba.load_userdict("userdict.txt")
# stopwords = set(open('stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))   #读入停用词
raw_documents = [
    '0无偿居间介绍买卖毒品的行为应如何定性',
    '1吸毒男动态持有大量毒品的行为该如何认定',
    '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '5为获报酬帮人购买毒品的行为该如何认定',
    '6毒贩出狱后再次够买毒品途中被抓的行为认定',
    '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '9一方未签字办理的结婚登记是否有效',
    '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '12身份证被别人冒用无法登记结婚怎么办？',
    '13同居后又与他人登记结婚是否构成重婚罪',
    '14未办登记只举办结婚仪式可起诉离婚吗',
    '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]
corpora_documents = []
for item_text in raw_documents:
    item_str = jieba.lcut(item_text)
    print(item_str)
    corpora_documents.append(item_str)
print(corpora_documents)
# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
corpus = [dictionary.doc2bow(text) for text in corpora_documents]
#num_features代表生成的向量的维数（根据词袋的大小来定）
similarity = Similarity('-Similarity-index', corpus, num_features=400)

test_data_1 = '你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚'
test_cut_raw_1 = jieba.lcut(test_data_1)

print(test_cut_raw_1)
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 5
print(similarity[test_corpus_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples

print('################################')

test_data_2 = '家人因涉嫌运输毒品被抓，她只是去朋友家探望朋友的，结果就被抓了，还在朋友家收出毒品，可家人的身上和行李中都没有。现在已经拘留10多天了，请问会被判刑吗'
test_cut_raw_2 = jieba.lcut(test_data_2)
print(test_cut_raw_2)
test_corpus_2 = dictionary.doc2bow(test_cut_raw_2)
similarity.num_best = 5
print(similarity[test_corpus_2])

# # 利用jieba分割句子
# def cutSentences(text):
#      newText = [jieba.lcut(line.replace("\n", "")) for line in text]
#      return newText
#
# def loadData(path):
#     contents = pd.read_csv(path + "weibo_senti_100k.csv")
#     labels = contents["label"]
#     sentences = contents["review"]
#     return sentences, labels
#
# def divideDatasetsFromCSV(path):
#     sentences = pd.read_csv(path + "weibo_senti_100k.csv")
#     sentences_num = sentences.shape[0]
#     pos_sentences_num = sentences[sentences.label == 1].shape[0]
#     neg_sentences_num = sentences[sentences.label == 0].shape[0]
#     pos_sentences = sentences[sentences.label == 1]['review']
#     neg_sentences = sentences[sentences.label == 0]['review']
#     pos_list = pos_sentences.tolist()
#     neg_list = neg_sentences.tolist()
# #   打乱列表
#     random.shuffle(pos_list)
#     random.shuffle(neg_list)
# #   划分训练验证测试
#     train_sentences = pos_list[0:pos_sentences_num*TRAIN_RATE]
#     train_sentences.extend(neg_list[0:neg_sentences_num*TRAIN_RATE])
#     val_sentences = pos_list[pos_sentences_num*TRAIN_RATE:pos_sentences_num*(TRAIN_RATE+VAL_RATE)]
#     val_sentences.extend(neg_list[neg_sentences_num*TRAIN_RATE:neg_sentences_num*(TRAIN_RATE+VAL_RATE)])
#     test_sentences = pos_list[pos_sentences_num*(TRAIN_RATE+VAL_RATE):pos_sentences_num]
#     test_sentences.extend(neg_list[neg_sentences_num*(TRAIN_RATE+VAL_RATE):neg_sentences_num])
#     return train_sentences, val_sentences, test_sentences
#
# def buildDictionary(model, sentences):
#     gensim_dic = Dictionary()
#     # 划分词袋
#     # print(model.wv.index_to_key)
#     gensim_dic.doc2bow(model.wv.index_to_key, allow_update=True)
#     # 寻找词频数超过10的索引和向量
#     w2index = {v: k+1 for k, v in gensim_dic.items()}
#     w2vec = {word: model.wv[word] for word in w2index.keys()}
#
#     def parseDatasets(sentences):
#         data = []
#         for sentence in sentences:
#             new_txt = []
#             for word in sentence:
#                 try:
#                     new_txt.append(w2index(word))
#                 except:
#                     new_txt.append(0)
#             data.append(new_txt)
#         return data
#     parse_sent = parseDatasets(sentences)
#     parse_sent = sequence.pad_sequences(parse_sent, maxlen=maxlen)
#     return w2index, w2vec, parse_sent
#
# def buildTrainVec(sentences):
#     model = Word2Vec(vector_size=vocab_dim,
#                      min_count=n_exposures,
#                      window=window_size,
#                      workers=cpu_count,
#                      epochs=n_iterations)
#     model.build_vocab(sentences)
#     model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
#     model.save('./lstm_data/word2vec.pkl')
#     dic_indexes, word_vectors, parse_sent = buildDictionary(model=model, sentences=sentences)
#     return dic_indexes, word_vectors, parse_sent