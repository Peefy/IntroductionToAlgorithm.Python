
# python test/chapter1/chapter1test.py 
# python3 test/chapter1/chapter1test.py 

from __future__ import division, print_function, absolute_import
import unittest

class TestStringMethods(unittest.TestCase):
 
    def setUp(self):
        # Do something to initiate the test environment here.
        pass
 
    def tearDown(self):
        # Do something to clear the test environment here.
        pass
 
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')
 
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
 
if __name__ == '__main__':
    unittest.main()

# python test/chapter1/chapter1test.py 
# python3 test/chapter1/chapter1test.py 
