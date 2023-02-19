import unittest
from sample import core

class AddNumTest(unittest.TestCase):
    def test_add_num(self):
        expect = 3
        actual = core.add_num(1, 2)
        self.assertEqual(expect, actual)