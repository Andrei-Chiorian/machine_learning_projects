import sys
import time
import threading


class Spinner:
    def __init__(self):
        self.spinner = ["|", "/", "-", "\\"]
        self.index = 0
        self.active = True
        self.text = 'Loading'
        self.thread = None
        self.OKGREEN = '\033[92m'
        self.ENDC = '\033[0m'
        self.OKCYAN = '\033[96m'

    def start(self):
        def spin():
            while self.active:
                sys.stdout.write('\r' + self.OKGREEN + 'Loading ' + self.ENDC + self.OKCYAN + self.text +
                                 self.ENDC + self.OKGREEN + '--> ' + self.spinner[self.index] + self.ENDC)
                sys.stdout.flush()
                self.index = (self.index + 1) % len(self.spinner)
                time.sleep(0.1)
        self.thread = threading.Thread(target=spin)
        self.thread.start()

    def stop(self):
        self.active = False
        self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.text) + 10) + '\r')

    def run_with_spinner(self, func, text='program'):
        self.text = text
        self.index = 0
        self.active = True
        self.start()
        result = func()
        self.stop()
        return result

