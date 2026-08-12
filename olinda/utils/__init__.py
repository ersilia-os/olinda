"""Small cross-cutting utilities.

Deliberately empty of imports. `olinda --help` costs one `rich_click` import today, and re-exporting
:mod:`olinda.utils.logging` from here would make any future ``from olinda.utils.x import y`` drag
loguru, rich and :mod:`olinda.console` in behind it. Import the submodule you want.
"""
