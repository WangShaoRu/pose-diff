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

    def distance(self):
        return np.linalg.norm(self.pt1[:2] - self.pt2[:2])

    def normalize(self):
        dis = self.distance()
        # TODO
        assert dis > 10
        return (self.pt2 - self.pt1) / dis, dis


class Node(object):
    def __init__(self, idx, score):
        self.idx = idx
        self.score = score