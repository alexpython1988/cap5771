from naive_bayes import MyGaussianNB
from naive_bayes import CategoricalNB
import unittest
from sklearn import datasets
import numpy as np

class TestNaiveBayes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_set = [(1,), (10,), (3,), (100,), (500,), (30,), (5,), (2,), (20,), (90,), (8,)]
        cls.train_label = ['X', 'Y', 'X', 'Z', 'Z', 'Y', 'X', 'X', 'Y', 'Y', 'X']
        cls.test_set = [(6,), (80,), (300,)]

    def setUp(self):
        self.GNB = MyGaussianNB()
        self.CNB = CategoricalNB()

    def test010(self):
        self.GNB.train(self.train_set, self.train_label)
        pred = self.GNB.predict(self.test_set)
        print(pred)

        self.CNB.train(self.train_set, self.train_label)
        pred = self.CNB.predict(self.test_set)
        print(pred)

    def test020(self):
        iris = datasets.load_iris()
        l = len(iris.data)
        dataset = list(zip(iris.data, iris.target))
        np.random.shuffle(dataset)
        train_set = [dataset[x][0] for x in range(100)]
        train_label = [dataset[x][1] for x in range(100)]
        test_set = [dataset[x][0] for x in range(100,120)]
        test_label = [dataset[x][1] for x in range(100, 120)]
        self.GNB.train(train_set, train_label)
        pred = self.GNB.predict(test_set)
        print(pred)
        print(test_label)

        self.CNB.train(train_set, train_label)
        pred = self.CNB.predict(test_set)
        print(pred)
        print(test_label)


if __name__ == '__main__':
    unittest.main()