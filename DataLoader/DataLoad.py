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


# Test code

# data = CSVReader("test.csv")

# print(data.header)
# for rowdata in data.reader:
#    for info in rowdata:
#         print(info, end=' ')
#     print("")
