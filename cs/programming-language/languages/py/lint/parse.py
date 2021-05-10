"""
pep8 parse.py
"""

from __future__ import print_function
import logging


class Command(ScrapyCommand):

    @property
    def max_evel(self):
        levels = self.items.keys() + self.requests.keys()
        if levels:
            return max(levels)
        else:
            return 0

    def add_items(self, lvl, new_items):
        old_items = self.items.get(lvl, [])
        self.items[lvl] = old_items + new_items
