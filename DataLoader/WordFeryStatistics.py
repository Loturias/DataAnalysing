# For Word frequency statistics
# One-Time script!

import DataLoader as DL

words = {}
read = DL.CSVReader("Data\\test.csv")
wordcount = 0


def CheckBlank(strs):
    if strs == "" or strs == " ":
        return False
    else:
        return True


for line in read.reader:
    texttemp = line[2]  # Get the text of this row
    sequtemp = texttemp.split(".")  # Cut by "."
    for sequ in sequtemp:
        if CheckBlank(sequ):
            texttemp = sequ.split(",")  # Cut by ","
            for text in texttemp:
                if CheckBlank(text):
                    wordtemp = text.split(" ")  # Finally,cut by " "
                    for word in wordtemp:
                        if CheckBlank(word):
                            low = word.lower()
                            if low not in words:
                                words[low] = 0
                                wordcount = wordcount + 1
                            else:
                                words[low] = words[low] + 1

print("Count Complete.Word count:"+str(wordcount))
