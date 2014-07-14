__author__ = 'davide'

import unittest
from labyrinth.labyrinth import Labyrinth


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.lab = Labyrinth(5, 6)

    def test_contains(self):
        for i in range(-10, 10):
            for j in range(-10, 10):
                if (0 <= i < self.lab.w and
                    0 <= j < self.lab.h):
                    self.assertTrue((i, j) in self.lab)
                else:
                    self.assertFalse((i, j) in self.lab)

    def test_set_item(self):
        self.assertTrue(self.lab[4, 2])
        self.lab[4, 2] = False
        self.assertFalse(self.lab[4, 2])


if __name__ == '__main__':
    unittest.main()
