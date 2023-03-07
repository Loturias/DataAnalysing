# Text processer


import DataLoader.DataLoad as dL
import LogSys.LogSys as Log
import json

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
            proword = word.replace(",", "").replace(".", "").replace("?","")
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
    def GetTextBatch(self):
        row = (next(self.reader.reader))
        text = row[2]
        texteval = Cour_Eff[row[4]]  # 0,1,2
        TextVec = [0 for i in range(self.textMaxLen)]  # Set the max length to 200
        count = 0

        RawTextVec = text.lower().replace("聽", " ").split(" ")  # Remove the invalid char
        for word in RawTextVec:
            proword = word.strip(".").strip(",")
            if count >= 200:
                break
            if proword in self.wordlib:
                TextVec[count] = self.wordlib[proword][1]
                count = count + 1
        return TextVec, texteval


# Test code

# lib = json.load(open("WordLib.json", "r"))
# test = TSetApp(lib)
# for i in range(10):
#     batch = test.GetTextBatch()
#    print(batch)

