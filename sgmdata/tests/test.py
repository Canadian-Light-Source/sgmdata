import unittest
from sgmdata import SGMData
from sgmdata import SGMQuery

class TestLoad(unittest.TestCase):
    sgmq = SGMQuery(sample="GeO2", proposal="35C12468", load=False)

    def test_search(self):
        self.assertTrue(len(self.sgmq.paths))

    def test_load(self):
        self.data = SGMData(self.sgmq.paths)
        self.assertTrue(len(self.data.scans))

    def test_interp(self):
        interp = self.data.interpolate()
        self.assertTrue(len(interp))

    def test_avg(self):
        avg = self.data.mean()
        self.assertTrue(len(avg) and hasattr(self.data, 'averaged'))

if __name__ == '__main__':
    unittest.main()