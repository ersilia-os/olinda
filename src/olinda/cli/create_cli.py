from .cmd import Command
from .commands import olinda_cli

def create_cli():
    cmd = Command()
    cmd.distill()
    cmd.predict()
    
    return olinda_cli
