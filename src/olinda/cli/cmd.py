from .commands.distill import distill_cmd

class Command(object):
    def __init__(self):
        pass

    def distill(self):
        distill_cmd()