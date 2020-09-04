import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import display_pose
from core import PoseDiff
from core.pose_25 import VIS_KP
from core import PoseException


if __name__ == '__main__':
    ref_pose = np.load('pose5.npy')
    test_pose = np.load('pose4.npy')
    ref_pose = ref_pose[VIS_KP]
    test_pose = test_pose[VIS_KP]
    ref_img = cv2.imread("input/test5.jpg")
    test_img = cv2.imread("input/test4.jpg")
    ref_img_with_pose = display_pose(ref_img, ref_pose, color=(0, 255, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_img_with_pose[:, :, ::-1])
    pd = PoseDiff()
    try:
        scores = pd.pose_diff(ref_pose, test_pose)
    except PoseException as e:
        print(e)
        scores = None
    test_img_with_pose = display_pose(
        test_img, test_pose, color=(0, 255, 255), scores=scores)
    plt.subplot(1, 2, 2)
    plt.imshow(test_img_with_pose[:, :, ::-1])
    plt.show()
    t = 0
