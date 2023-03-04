# For Word frequency statistics
# One-Time script!

import DataLoader as DL
import LogSys.LogSys as Log

words = {}
read = DL.CSVReader("Data\\train.csv")
wordcount = 0

wordNum = 1

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
                        intemp = word.split("聽")  # Add a more filter to remove the NBSP character
                        for ins in intemp:
                            if CheckBlank(ins):
                                low = ins.lower()
                                if ins not in words:
                                    words[low] = 1
                                    wordcount = wordcount + 1
                                else:
                                    words[low] = words[low] + 1

Log.Print("Count Complete.Word count:"+str(wordcount))
ValidWord = {}
for key in words:
    if (key.isalpha() or ("-" in key.strip("-"))) and words[key] > 1:
        ValidWord[key] = words[key]
    # else:
    #     InWord[key] = words[key]
Log.Print("Valid word count: "+str(len(ValidWord)))


ValidLis = []
WordCount = 0
for key in ValidWord:
    ValidLis.append({key: [ValidWord[key], WordCount]})
    WordCount = WordCount + 1

print(ValidLis)
# Train Data's valid word count:12271
