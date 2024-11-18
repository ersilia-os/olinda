import click
from ... import __version__

@click.group(cls=click.Group)
@click.version_option(version=__version__)

def olinda_cli():
    """
    Olinda: An automated tool for QSAR model distillation
    """
