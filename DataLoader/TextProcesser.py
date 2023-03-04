# Text processer


import DataLoader as dL
import LogSys.LogSys as Log
import json

class TSetApp:
    def __init__(self, WordLib):
        self.BatchTextSize = 32   # The biggest size of a batch(count by raw text)
        self.squeMaxLen = 64  # The biggest sentence length indata
        self.filepath = "Data\\train.csv"
        self.reader = dL.CSVReader(self.filepath)
        self.wordlib = WordLib

    def SetBatchSize(self, size):  # Set the Batch's max size
        self.BatchTextSize = size

    def GetNextSenteVec(self, seq):  # This function return an integer sequence of a sentence
        rawseq = [0 for _ in range(self.squeMaxLen)]
        count = 0
        for word in seq.lower().split():
            proword = word.replace(",", "").replace(".", "").replace("?","")
            if proword in self.wordlib:
                try:
                    rawseq[count] = self.wordlib[proword][1]
                except:
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


# Test code

# lib = json.load(open("WordLib.json", "r"))
# test = TSetApp(lib)
# for i in range(100):
#     batch = test.GetEmbedBatch()


