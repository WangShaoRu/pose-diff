import numpy as np
import cv2
from .pose_25 import VIS_KP, LIMBS, KP_TREE
from .core import Pair, Node
from .pose_exception import BodyTooShortException


class PoseDiff(object):
    def __init__(self):
        self.body_min_distance = 20
        self._eps = 1e-6

    def pose_diff(self, ref_pose, test_pose):
        match_state = [0] * len(VIS_KP)
        # align ref_pose to test_pose by body
        ref_body_len = self._distance(ref_pose[1, :2], ref_pose[8, :2])
        if  ref_body_len < self.body_min_distance:
            raise BodyTooShortException('reference', ref_body_len)
        test_body_len = self._distance(test_pose[1, :2], test_pose[8, :2])
        if test_body_len < self.body_min_distance:
            raise BodyTooShortException('test', test_body_len)
        ref_pose = self._align(ref_pose, test_pose)

        queue = []
        queue.append(Node(8, 1))
        while queue:
            node = queue.pop()
            begin = node.idx
            match_state[begin] = node.score
            if begin in KP_TREE:
                for end in KP_TREE[begin]:
                    limb_ref = Pair(ref_pose[begin], ref_pose[end])
                    limb_test = Pair(test_pose[begin], test_pose[end])
                    score = self._match_limb(limb_ref, limb_test)
                    queue.append(Node(end, score * match_state[begin]))
        return match_state

    def _distance(self, pt1, pt2):
        return np.linalg.norm(pt1 - pt2)

    def _align(self, ref_pose, test_pose):
        body_ref = Pair(ref_pose[8][:2], ref_pose[1][:2])
        body_test = Pair(test_pose[8][:2], test_pose[1][:2])
        scale_ratio = body_test.distance() / (body_ref.distance() + self._eps)
        shift = body_test.pt1 - scale_ratio * body_ref.pt1
        ref_pose = np.concatenate(
            (scale_ratio * ref_pose[:, :2] + shift[None, :], ref_pose[:, 1:2]), axis=1)
        return ref_pose

    def _match_limb(self, limb_ref, limb_test):
        ref_vec, ref_len = limb_ref.normalize()
        test_vec, test_len = limb_test.normalize()
        cos_score = (ref_vec * test_vec).sum()
        dis_score = min(ref_len / test_len, test_len / ref_len)
        return cos_score * dis_score