# This is the inter subscript of whole project

import os

import LogSys.LogSys as LogSys
import json
import DataLoader.TextProcesser as TAPP
import numpy as np

# 初始化PyTorch框架
LogSys.Print("Initializing PyTorch...")
import torch
import torch.nn as nn

LogSys.Print("PyTorch Version: " + torch.__version__)
LogSys.Print("CUDA Version: " + torch.version.cuda)
LogSys.Print("Is CUDA available: " + str(torch.cuda.is_available()))
LogSys.Print("GPU Device Info: " + torch.cuda.get_device_name(0))
torch.device = "cuda"

# 载入词典
fil = open(r"DataLoader\WordLib.json", "r")
WordLib = json.load(fil)

# 创建训练集管理器
Set = TAPP.TSetApp(WordLib, r"DataLoader\Data\train.csv")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 设置随机数种子，先辈保佑我准确率呀！
setup_seed(114514)

# Model definition
class TextClassifierRNN(nn.Module):
    def __init__(self, batchsize=10):
        super(TextClassifierRNN, self).__init__()

        # 设置嵌入层维度,单个训练集大小,输出层大小,rNN网络层数,隐藏层尺寸
        self.embedsize = 64
        self.batchsize = batchsize
        self.outsize = 3
        self.rnnlayers = 1
        self.hiddensize = self.embedsize

        # 设置嵌入层。
        self.embed = nn.Embedding(len(WordLib), self.embedsize)

        # 使用PyTorch封装的GRU单元作为rNN层
        self.gru = nn.GRU(self.embedsize, self.hiddensize, batch_first=True)

        # 用一个线性层作为输出层
        self.out = nn.Linear(self.hiddensize*Set.textMaxLen, self.outsize)

    # 前向传播
    def forward(self, tem):
        # 生成一个Batch
        data = []
        FinalEval = []
        smax = nn.Softmax(dim=1)

        for i in range(self.batchsize):
            temp, Eval = Set.GetTextBatch()
            data.append(temp)
            Evals = [0, 0, 0]
            Evals[Eval] = 1
            FinalEval.append(Evals)

        dataTen = torch.LongTensor(data)

        embedData = self.embed(dataTen)  # 大小为10个文本*最大200个词向量*64维

        # 创建一个初始上下文给GRU使用
        hidden = torch.autograd.Variable(torch.zeros(self.rnnlayers, 10, self.hiddensize))

        output, hidden = self.gru(embedData, hidden)  # 10*200*64

        # 把维度拍到10*12800
        xoutput = output.flatten(1, 2)

        # 喂给线性层出来10*7的结果，再过一遍Softmax函数归一化
        FinalOutput = smax(self.out(xoutput))  # 变成了10*7

        print(FinalOutput)
        print(FinalEval)
        return FinalOutput, FinalEval

# 创建模型
test = TextClassifierRNN()

