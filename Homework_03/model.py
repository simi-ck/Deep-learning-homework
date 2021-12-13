import torch
from torch import nn


def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    # >0.5为无恶意
    # <0.5为无恶意
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout = 0.5, fix_embedding = True):
        super(LSTM_Net, self).__init__()
        # 设计 embedding layer
        # torch.nn.Embedding:A simple lookup table that stores embeddings of a fixed dictionary and size.
        # num_embeddings ：字典中词的个数
        # embedding_dim：embedding的维度
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1)) # 单词数量*词向量维数
        # Embedding.weight (Tensor) – the learnable weights of the module of shape
        # embedding即为之前word2vec训练出来的词向量矩阵
        self.embedding.weight = torch.nn.Parameter(embedding) # nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去
        # 是否将embedding固定
        # 如果fix_embedding为False,在训练过程中，embedding也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first=True)
        # nn.LSTM部分参数说明
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        # dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())

    def forward(self, inputs):
        # 这一步根据输入的数据的idx，完成句子的embedding操作
        inputs = self.embedding(inputs)
        # inputs的dimension由(batch, seq_len)变为(batch, seq_len, input_size) 128*30*250
        x, _ = self.lstm(inputs, None)
        # x的dimension (batch, seq_len, num_directions * hidden_size) 128*30*300
        # hidden_size:The number of features in the hidden state h
        # 取LSTM最后一层的hidden state
        x = x[:, -1, :] # 128*300, 注意这里会直接降一维
        x = self.classifier(x) # 128*1
        return x

