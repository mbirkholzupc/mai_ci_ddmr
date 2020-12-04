import os
import re

class config_file:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            cfg_txt = f.read()

        # Add some sane defaults
        self.indatadir = ''
        self.classes = []
        self.outdatadir = ''
        self.model = ''
        self.augment = False

        # Parse config file
        self.parse(cfg_txt)

    def parse(self, cfg_str):
        match = re.search('^indatadir=(.+)$', cfg_str, flags=re.MULTILINE)
        if match:
            self.indatadir = match.group(1)
        self.classes = [d for d in os.listdir(self.indatadir)]

        match = re.search('^outdatadir=(.+)$', cfg_str, flags=re.MULTILINE)
        if match:
            self.outdatadir = match.group(1)
        if not os.path.exists(self.outdatadir):
            os.mkdir(self.outdatadir, mode=0o755)

        match = re.search('^model=([a-z]+)$', cfg_str, flags=re.MULTILINE)
        if match:
            self.model = match.group(1)

        match = re.search('^augment=([a-z]+)$', cfg_str, flags=re.MULTILINE)
        if match:
            self.augment = True if match.group(1) == 'true' else False

    def __str__(self):
        data  = f"indatadir: {self.indatadir}\n"
        data += f"outdatadir: {self.outdatadir}\n"
        data += f"classes: {self.classes}\n"
        data += f"model: {self.model}\n"
        data += f"augment: {self.augment}\n"
        
        return data


