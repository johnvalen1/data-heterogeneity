

def logText(txt):
    """
    Use this function to log text from output to a text file for later revision.
    """

    log_dir = './log/'
    log_file_name = 'info_log.txt'
    with open(log_dir + log_file_name, 'a') as f:
        print(txt, file=f)
