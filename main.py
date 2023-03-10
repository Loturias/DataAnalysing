# This is the inter subscript of whole project

import os
import random

import DataLoader.TextProcesser
import LogSys.LogSys as LogSys
import json
import DataLoader.TextProcesser as TAPP
import numpy as np
import datetime as dt
import time

TextType = {0: 'Lead',  1: 'Position', 2: 'Claim', 3: 'Evidence' ,
            4:'Counterclaim', 5:'Rebuttal', 6: 'Concluding Statement'}

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

# 测试集管理器

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 设置随机数种子，先辈保佑我准确率呀！

se = 114514
LogSys.Print("Random seed: " + str(se))
setup_seed(se)

# Model definition
class TextClassifierRNN(nn.Module):
    def __init__(self, Set, batchsize=35):
        super(TextClassifierRNN, self).__init__()

        # 设置嵌入层维度,单个训练集大小,输出层大小,rNN网络层数,隐藏层尺寸
        self.embedsize = 64
        self.batchsize = batchsize
        self.outsize = 7
        self.rnnlayers = 1
        self.hiddensize = self.embedsize

        # 设置嵌入层。
        self.embed = nn.Embedding(len(WordLib), self.embedsize)

        self.dp1 = torch.nn.ReLU()

        # 使用PyTorch封装的GRU单元作为rNN层
        self.gru = nn.GRU(self.embedsize, self.hiddensize, batch_first=True, num_layers=self.rnnlayers)
        #self.lstm = nn.LSTM(self.embedsize, self.hiddensize, batch_first=True)

        self.dp2 = torch.nn.RReLU()

        # 用一个线性层作为输出层
        self.out = nn.Linear(self.hiddensize*Set.textMaxLen, self.outsize)

    # 前向传播
    def forward(self, dataTen, Eval):
        smax = nn.Softmax(dim=1)
        try:
            embedData = self.embed(dataTen)  # 大小为10个文本*最大200个词向量*64维
        except:
            return
        # 创建一个初始上下文给GRU使用
        hidden = torch.autograd.Variable(torch.zeros(self.rnnlayers, self.batchsize, self.hiddensize))

        output, hidden = self.gru(embedData, hidden)  # 10*200*64

        # output = self.dp1(output)
        # 把维度拍到10*12800
        xoutput = output.flatten(1, 2)

        xoutput = self.dp2(xoutput)

        xoutput = self.out(xoutput)
        # 喂给线性层出来10*7的结果，再过一遍Softmax函数归一化
        FinalOutput = smax(xoutput)  # 变成了10*7

        return FinalOutput, Eval

def TrainModel(Model,TSet):
    Model.train()
    # 指定损失函数
    # 查了一下建议rNN用交叉熵损失函数，遂查文档抄之
    lossfunc = torch.nn.CrossEntropyLoss()

    # 然后是优化器
    # 都拆成10个一块的小Batch了当然用SGD
    # 最后一个参数是个加权优化项，虽然是可选项但是先填个0.9进去看看效果
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.06, momentum=0.9)

    LogSys.Print("Train Begin...")
    sttime = time.time()
    # int(TSet.reader.GetRowCount()/10)
    for i in range(int(TSet.reader.GetRowCount(IsTrain=True)/Model.batchsize)):
        # 生成Batch的代码
        data = []
        Eval = []  # 标签数据

        for x in range(Model.batchsize):
            temp, EvalNum = TSet.GetTextBatch(IsTrain=True)
            data.append(temp)
            tempeval = [0 for _ in range(7)]
            tempeval[EvalNum] = 1
            Eval.append(tempeval)

        dataTen = torch.LongTensor(data)  # 序列向量数据
        EvalTen = torch.Tensor(Eval)

        Output, Eval = Model(dataTen, EvalTen)  # 前向传播得到预测数据

        optimizer.zero_grad()
        loss = lossfunc(Output, Eval)
        loss.backward()
        optimizer.step()

        loss = loss.expand(1)

        # 打印下降程度
        LogSys.Print("Step {} Loss: ".format(i+1) + str(round(float(loss.data[0]), 4)))

    LogSys.Print("Train End")
    endtime = time.time()
    LogSys.Print("Training time:{}".format(round(endtime-sttime, 2)))

def TestModel(Model,TestSet):
    # 关闭计算梯度，设置为测试模式
    with torch.no_grad():
        Model.eval()
        Model.batchsize = 10
        # TestSet.reader.GetRowCount()
        for i in range(1):
            data = []
            Eval = []  # 标签数据
            for k in range(10):
                temp, EvalNum = TestSet.GetTextBatch(IsTrain=False)
                data.append(temp)
                tempeval = [0 for i in range(7)]
                tempeval[EvalNum] = 1
                Eval.append(tempeval)

            dataTen = torch.LongTensor(data)
            EvalTen = torch.Tensor(Eval)

            Output, Eval = Model(dataTen, EvalTen)  # Output为3维向量

            Su = 0.0
            for x in range(10):
                count = 0
                for k in range(7):
                    if int(Eval[x][k]) == 1:
                        count = k
                LogSys.Print("Text {}'s classify accuracy: ".format(x+1)+str(round(float(Output[x][count]), 4))+" Text Type: "+TextType[count])
                Su = Su + round(float(Output[x][count]), 4)

            #print(Output)
            LogSys.Print("Average accuracy: " + str(round(Su/10, 2)))
            localtime = time.localtime(time.time())
            LogSys.Print("Training Date:{}/{} {}:{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        return round(Su/10, 2)

# 创建模型
device = "cuda"
LogSys.Print("Training Device: "+device)

Set = TAPP.TSetApp(WordLib, r"DataLoader\Data\final_train.csv")
TestSet = TAPP.TSetApp(WordLib, r"DataLoader\Data\test.csv")

test = TextClassifierRNN(Set)

TrainModel(test, Set)
rate = TestModel(test, TestSet)




