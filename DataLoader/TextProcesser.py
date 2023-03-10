# Text processer
import random

import DataLoader.DataLoad as dL
import LogSys.LogSys as Log
import json
import csv

Cour_Eff = {"Effective": 0, "Adequate": 1, "Ineffective": 2}

# Hard coded here!
TextType = {'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3,
            'Counterclaim': 4, 'Rebuttal': 5, 'Concluding Statement': 6}

chr_dict = [chr(i) for i in range(ord('a'), ord('z') + 1)]
class TSetApp:
    def __init__(self, WordLib, filepath):
        self.BatchTextSize = 32   # The biggest size of a batch(count by raw text)
        self.squeMaxLen = 64  # The biggest sentence length indata
        self.textMaxLen = 200
        self.filepath = filepath
        self.reader = dL.CSVReader(self.filepath)
        self.wordlib = WordLib

    def SetBatchSize(self, size):  # Set the Batch's max size
        self.BatchTextSize = size

    def GetNextSenteVec(self, seq):  # This function return an integer sequence of a sentence
        rawseq = [0 for _ in range(self.squeMaxLen)]
        count = 0
        for word in seq.lower().split():
            proword = word.replace(",", "").replace(".", "").replace("?"," ")
            if proword in self.wordlib:
                try:
                    rawseq[count] = self.wordlib[proword][1]
                except:
                    Log.Print("Buffer Overflow.Surplus data is discarded")
                    return []  # Throw sentences longer than 64
                count = count + 1
        return rawseq

    def GetEmbedBatch(self):
        Batch = []
        ccount = 0
        while ccount < self.BatchTextSize:
            rawseq = []
            try:
                rawseq = (next(self.reader.reader))[2]
            except StopIteration:
                break
            sequ = []
            for sent in rawseq.split("."):
                if sent != "":
                    sequ = self.GetNextSenteVec(sent)
                if sequ:
                    if sequ[0] != 0:
                        Batch.append(sequ)
                        ccount = ccount + 1

        return Batch

    # Return a vector composed by word's raw code(Need processed by embedding layer
    def GetTextBatch(self, IsTrain):
        try:
            row = 0
            if IsTrain:
                row = (next(self.reader.reader))
                while row[4] == "Ineffective":
                    row = (next(self.reader.reader))
            else:
                row = (next(self.reader.reader))
            text = row[2]
            texteval = TextType[row[3]]  # 0,1,2
            TextVec = [0 for i in range(self.textMaxLen)]  # Set the max length to 200
            count = 0
        except StopIteration:
            return

        RawTextVec = text.lower().replace("聽", " ").split(" ")  # Remove the invalid char
        for word in RawTextVec:
            proword = word.strip(".").strip(",")
            if count >= self.textMaxLen:
                break
            if proword in self.wordlib:
                TextVec[count] = self.wordlib[proword][1]
                count = count + 1
        return TextVec, texteval

    def CheckTextDistri(self):
        TextTypeIn = {Type: 0 for Type in TextType}
        count = self.reader.GetRowCount(IsTrain=False)
        self.reader.SetDefault()
        for i in range(count):
            ty = next(self.reader.reader)[3]
            TextTypeIn[ty] = TextTypeIn[ty] + 1

        Num = sum(TextTypeIn[typ] for typ in TextTypeIn)
        for typs in TextTypeIn:
            print("Type "+typs+":"+str(round(TextTypeIn[typs]/Num,3)))
        print(TextTypeIn)
# Test code

# lib = json.load(open("WordLib.json", "r"))
# test = TSetApp(lib, "Data\\train.csv")
# test.CheckTextDistri()


"""
Type Lead:0.062
Type Position:0.109
Type Claim:0.326
Type Evidence:0.329
Type Counterclaim:0.048
Type Rebuttal:0.034
Type Concluding Statement:0.091
"""


def ClearData(lib):
    TypeMax = {'Lead': 1000, 'Position': 1000, 'Claim': 1000, 'Evidence': 1000,
            'Counterclaim': 1000, 'Rebuttal': 1000, 'Concluding Statement': 1000}

    f = open("Data\\clear_train.csv", "w", newline='')
    RawTrain = TSetApp(lib, "Data\\train.csv")
    writer = csv.writer(f)

    writer.writerow(["discourse_id", "essay_id",
                     "discourse_text", "discourse_type", "discourse_effectiveness"])

    # 以最小的Rebuttal为基准，每组数据准备1000个
    TypeCount = {key: 0 for key in TextType}

    for row in RawTrain.reader.reader:
        if TypeCount[row[3]] < TypeMax[row[3]] and row[4] != "Ineffective":
            writer.writerow(row)
            TypeCount[row[3]] = TypeCount[row[3]] + 1
    # 分布随机化
    f.close()


def DataRandomization(lib):
    fp = open("Data\\final_train.csv", "w", newline='')
    fpwriter = csv.writer(fp)
    fpwriter.writerow(["discourse_id", "essay_id",
                     "discourse_text", "discourse_type", "discourse_effectiveness"])

    RawTrain = TSetApp(lib, "Data\\clear_train.csv")
    datalis = list(RawTrain.reader.reader)

    maxlen = len(datalis)
    for i in range(maxlen):
        count = random.randint(0, len(datalis)-1)
        fpwriter.writerow(datalis[count])
        datalis.pop(count)

    fp.close()



# lib = json.load(open("WordLib.json", "r"))
# ClearData(lib)
# DataRandomization(lib)
# test2 = TSetApp(lib, "Data\\final_train.csv")
# test2.CheckTextDistri()
# Log.Print("Data Refreshed.")
