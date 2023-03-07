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

# 测试集管理器
TestSet = TAPP.TSetApp(WordLib, r"DataLoader\Data\test.csv")

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
        self.outsize = 7
        self.rnnlayers = 1
        self.hiddensize = self.embedsize

        # 设置嵌入层。
        self.embed = nn.Embedding(len(WordLib), self.embedsize)

        # 使用PyTorch封装的GRU单元作为rNN层
        self.gru = nn.GRU(self.embedsize, self.hiddensize, batch_first=True)

        # 用一个线性层作为输出层
        self.out = nn.Linear(self.hiddensize*Set.textMaxLen, self.outsize)

    # 前向传播
    def forward(self, dataTen, Eval):
        smax = nn.Softmax(dim=1)
        embedData = self.embed(dataTen)  # 大小为10个文本*最大200个词向量*64维

        # 创建一个初始上下文给GRU使用
        hidden = torch.autograd.Variable(torch.zeros(self.rnnlayers, 10, self.hiddensize))

        output, hidden = self.gru(embedData, hidden)  # 10*200*64

        # 把维度拍到10*12800
        xoutput = output.flatten(1, 2)

        # 喂给线性层出来10*7的结果，再过一遍Softmax函数归一化
        FinalOutput = smax(self.out(xoutput))  # 变成了10*7

        return FinalOutput, Eval

def TrainModel(Model,TSet):
    Model.train()
    # 指定损失函数
    # 查了一下建议rNN用交叉熵损失函数，遂查文档抄之
    lossfunc = torch.nn.CrossEntropyLoss()

    # 然后是优化器
    # 都拆成10个一块的小Batch了当然用SGD
    # 最后一个参数是个加权优化项，虽然是可选项但是先填个0.9进去看看效果
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)

    LogSys.Print("Train Begin...")
    # int(TSet.reader.GetRowCount()/10)
    for i in range(1):
        # 生成Batch的代码
        data = []
        Eval = []  # 标签数据

        for i in range(Model.batchsize):
            temp, EvalNum = TSet.GetTextBatch()
            data.append(temp)
            tempeval = [0 for i in range(7)]
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
        LogSys.Print("Loss: " + str(round(float(loss.data[0]), 4)))

    LogSys.Print("Train End")


def TestModel(Model,TestSet):
    # 关闭计算梯度，设置为测试模式
    with torch.no_grad():
        Model.eval()
        # TestSet.reader.GetRowCount()
        for i in range(1):
            data = []
            Eval = []  # 标签数据
            for k in range(Model.batchsize):
                temp, EvalNum = TestSet.GetTextBatch()
                data.append(temp)
                tempeval = [0 for i in range(7)]
                tempeval[EvalNum] = 1
                Eval.append(tempeval)

            dataTen = torch.LongTensor(data)
            EvalTen = torch.Tensor(Eval)

            Output, Eval = Model(dataTen, EvalTen)  # Output为3维向量
            for x in range(Model.batchsize):
                count = 0
                for k in range(7):
                    if int(Eval[x][k]) == 1:
                        count = k
                LogSys.Print("Text {}'s classify accuracy: ".format(x+1)+str(round(float(Output[x][count]), 4)))



# 创建模型
test = TextClassifierRNN()
TrainModel(test, Set)
TestModel(test, TestSet)


