# For Reading Data from .csv file

import csv


class CSVReader:
    def __init__(self, FileName,):
        self.file = open(FileName, 'rt')
        self.reader = csv.reader(self.file)
        self.header = next(self.reader)

    def __del__(self):  # For file closing
        self.file.close()

    def ReadLine(self):  # Return the next line's data from CSV as tuple
        return tuple(next(self.reader))

    def GetRowCount(self, IsTrain):
        self.file.seek(0)
        next(self.reader)
        res = 0
        if IsTrain:
            for row in self.reader:
                if row[4] != "Ineffective":
                    res = res + 1
        else:
            res = sum(1 for _ in enumerate(self.reader)) - 1
        self.file.seek(0)
        next(self.reader)  # 重置索引到第一行
        print("Valid row count: " + str(res))
        return res

    def SetDefault(self):
        self.file.seek(0)
        next(self.reader)  # 重置索引到第一行
# Test code



# print(data.header)
# for rowdata in data.reader:
#    for info in rowdata:
#         print(info, end=' ')
#     print("")

