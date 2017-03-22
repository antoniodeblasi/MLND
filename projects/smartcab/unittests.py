'''
Created on Jan 19, 2017

@author: antoniodeblasi
'''
import unittest
from agent import ConstantAlpha, AverageAlpha, LinearDecayEpsilon


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testConstantAlpha(self):
        ca = ConstantAlpha(0.5)
        self.assertEqual(0.5, ca(), "")
        self.assertEqual(0.5, ca(), "")
    
    def testAverageAlpha(self):
        aa = AverageAlpha()
        self.assertAlmostEqual(1.0, aa())
        self.assertAlmostEqual(0.5, aa())
        self.assertAlmostEqual(1.0/3, aa())
        
    def testLinearDecayEpsilon(self):
        lde = LinearDecayEpsilon()
        self.assertAlmostEqual(1.0, lde())
        self.assertAlmostEqual(1.0 -1.0/2, lde())
        self.assertAlmostEqual(1.0 -1.0/2 -1.0/3, lde())
        self.assertAlmostEqual(1.0 -1.0/2 -1.0/3 -1.0/4, lde())
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()