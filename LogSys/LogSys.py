# Define simple Log API.

LogType = {
    0: "INFO",
    1: "WARNING",
    2: "ERROR"
}


def Print(Info,Type):
    print('['+LogType[Type]+']'+" "+Info)
