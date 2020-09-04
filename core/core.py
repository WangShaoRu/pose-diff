import numpy as np


class Pair(object):
    def __init__(self, pt1, pt2):
        assert isinstance(pt1, np.ndarray) and isinstance(
            pt2, np.ndarray), "pt should be numpy.ndarray!"
        assert pt1.shape == (2,) or pt1.shape == (
            3,), "pt should be with shape (2,) or (3,)"
        assert pt2.shape == (2,) or pt2.shape == (
            3,), "pt should be with shape (2,) or (3,)"
        self.pt1 = pt1
        self.pt2 = pt2
        self._eps = 1

    def distance(self):
        return np.linalg.norm(self.pt1[:2] - self.pt2[:2])

    def normalize(self):
        dis = self.distance()
        return (self.pt2 - self.pt1) / (dis + self._eps), dis

    def _check(self):
        return self._check_pt1(), self._check_pt2()

    def _check_pt1(self):
        return True if self.pt1.nonzero()[0].size else False

    def _check_pt2(self):
        return True if self.pt2.nonzero()[0].size else False


class Node(object):
    def __init__(self, idx, score):
        self.idx = idx
        self.score = score