import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import display_pose
from core import PoseDiff
from core.pose_25 import VIS_KP
from core import PoseException
import sys, os
import argparse
sys.path.append(os.path.join('./openpose/build/python'))


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Diff')
    parser.add_argument('--ref', type=str, default='./input/ref.jpg', help='Path to reference image')
    parser.add_argument('--test', type=str, default='./input/test.jpg', help='Path to test image')
    parser.add_argument('--show', action='store_true', help='Display')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    ref_img = cv2.imread(args.ref)
    test_img = cv2.imread(args.test)
    try:
        from openpose import pyopenpose as op
        params = dict()
        params['model_folder'] = os.path.join('./openpose/models')

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Reference Image and Display
        datum = op.Datum()
        
        datum.cvInputData = ref_img
        opWrapper.emplaceAndPop([datum])
        ref_pose = datum.poseKeypoints[0]

        # Process Test Image and Display
        datum.cvInputData = test_img
        opWrapper.emplaceAndPop([datum])
        test_pose = datum.poseKeypoints[0]
    except ImportError:
        print('Warning: openpose has not been properly installed! Pre-stored pose will be used!')
        ref_pose = np.load('input/ref_pose.npy')
        test_pose = np.load('input/test_pose.npy')
    except Exception as e:
        raise e

    ref_pose = ref_pose[VIS_KP]
    test_pose = test_pose[VIS_KP]

    ref_img_with_pose = display_pose(ref_img, ref_pose, color=(0, 255, 0))
    
    pd = PoseDiff()
    try:
        scores = pd.pose_diff(ref_pose, test_pose)
    except PoseException as e:
        print(e)
        scores = None
    test_img_with_pose = display_pose(
        test_img, test_pose, color=(0, 255, 255), scores=scores)
    
    if args.show:
        plt.subplot(1, 2, 1)
        plt.imshow(ref_img_with_pose[:, :, ::-1])
        plt.subplot(1, 2, 2)
        plt.imshow(test_img_with_pose[:, :, ::-1])
        plt.show()
    else:
        cv2.imwrite('./result/ref.jpg', ref_img_with_pose)
        cv2.imwrite('./result/test.jpg', test_img_with_pose)
