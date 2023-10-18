# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# # 选择模型，是使用哪个NLP模型
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# # 词向量维度
#
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()
class Args:
    def __init__(self):
        # self.model = 'Transformer'  # 选择模型
        # self.model = 'RNN'
        # self.model = 'TextCNN'
        self.model = 'RNN_variant'
        self.embedding = 'pre_trai  ned'  # 词向量维度
        self.word = True  # True表示按词分割，False表示按字符分割

args = Args()

if __name__ == '__main__':
    dataset = 'nlp-getting-started'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    # 预训练词向量

    if args.embedding == 'random':
        embedding = 'random'
    # 如果没有指定词向量维度则随机生成

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    # 加载模型，训练时选择


    from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    train(config, model, train_iter, dev_iter, test_iter)
