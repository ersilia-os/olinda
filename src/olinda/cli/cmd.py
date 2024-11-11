from .commands.distill import distill_cmd
from .commands.predict import predict_cmd

class Command(object):
    def __init__(self):
        pass

    def distill(self):
        distill_cmd()
        
    def predict(self):
        predict_cmd()
