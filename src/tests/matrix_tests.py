import unittest

from matrix import *


class TestStringMethods(unittest.TestCase):

    def test_transpose(self):
        a = [[1,2,3],
             [4,5,6]]
        transposed_a = [[1, 4],
                        [2, 5],
                        [3, 6]]

        self.assertListEqual(transpose(a), transposed_a)

    def test_multiply(self):
        a = [[1], [2]]
        b = [[2, 3]]
        b_a = [[8]]
        a_b = [[2, 3], [4, 6]]
        self.assertListEqual(multiply(b,a), b_a)
        self.assertListEqual(multiply(a,b), a_b)