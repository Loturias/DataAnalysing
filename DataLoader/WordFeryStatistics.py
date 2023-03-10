# For Word frequency statistics
# One-Time script!

import DataLoad as DL
import LogSys.LogSys as Log
import json

words = {}
read = DL.CSVReader("Data\\train.csv")
wordcount = 0

wordNum = 1
sentCount = 0
def CheckBlank(strs):
    if strs == "" or strs == " ":
        return False
    else:
        return True


type_dict = {}
typecount = 0

# Generate the dictionary
for line in read.reader:
    texttemp = line[2]  # Get the text of this row
    texttype = line[3]
    if texttype not in type_dict:
        type_dict[texttype] = typecount
        typecount = typecount + 1
    if "." not in texttemp:
        continue
    sequtemp = texttemp.split(".")  # Cut by "."
    for sequ in sequtemp:
        if CheckBlank(sequ):
            texttemp = sequ.split(",")  # Cut by ","
            for text in texttemp:
                if CheckBlank(text):
                    wordtemp = text.split(" ")  # Finally,cut by " "
                    # sentCount = max(sentCount, len(wordtemp))
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

Log.Print("Valid text type count: " + str(typecount))
print(type_dict)

ValidLis = {}
WordCount = 1
for key in ValidWord:
    ValidLis[key] = [ValidWord[key], WordCount]
    WordCount = WordCount + 1
# print(ValidLis)
# Train Data's valid word count:11783

json.dump(obj=ValidLis, fp=open("WordLib.json", 'w'))
Log.Print("Dictionary Saved")
