import sys
import json


class Logger:
    def __init__(self):
        self.cue = None
        self.print_real = sys.stdout
        self.running = False

    def print(self, string):
        self.print_real.write("\n" + str(string) + "\n")
        self.print_real.flush()

    def start(self, label):
        if not self.running:
            sys.stdout = Writer(label)
            self.cue = open("log/log_cue_" + str(label) + ".txt", "w")
            self.running = True
        else:
            self.print('Already running')

    def close(self):
        if self.running:
            sys.stdout.close()
            sys.stdout = self.print_real
            self.cue.close()
            self.running = False
        else:
            self.print('Nothing running')

    def write(self, data_dict):
        if self.running:
            print(json.dumps({"data": data_dict}))
        else:
            self.print('Nothing running')

    def cue_write(self, data_dict):
        if self.running:
            self.cue.write(json.dumps({"data": data_dict}) + '\n')
        else:
            self.print('Nothing running')


class Writer:
    def __init__(self, label):
        self.log = open("log/log_" + str(label) + ".txt", "w")

    def write(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()
