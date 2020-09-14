import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import display_pose
from core import PoseDiff
from core.pose_25 import VIS_KP
from core import PoseException
import sys
import os
import argparse
sys.path.append(os.path.join('./openpose/build/python'))


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Diff')
    parser.add_argument(
        '--ref', type=str, default='./input/ref.jpg', help='Path to reference image')
    parser.add_argument(
        '--test', type=str, default='./input/test.jpg', help='Path to test image/video')
    parser.add_argument(
        '--result', type=str, default='./output/', help='Path to restore the result video')
    parser.add_argument('--show', action='store_true', help='Display')
    args = parser.parse_args()

    return args


def pose_diff_img(ref_img, test_img):
    """
    distinguish the differences of the poses between the man in ref_img and test_img
    :param ref_img: np.ndarray, reference image
    :param test_img: np.ndarray, test image
    :return: ref_pose, test_pose, ref_img_with_pose, test_img_with_pose, scores
    """
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

    return ref_pose, test_pose, ref_img_with_pose, test_img_with_pose, scores


def pose_diff_video(ref_img, test_video, restore_path):
    """
    distinguish the differences of the poses between the man in ref_img and each frame of test_video
    :param ref_img: np.ndarray, reference image
    :param test_video: str, path to the test video
    :param restore_path: str, path to restore the result video
    :return: 
    """
    cap = cv2.VideoCapture(test_video)
    out = cv2.VideoWriter(restore_path,
                          cap.get(cv2.CAP_PROP_FOURCC),
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    use_openpose_flag = True
    try:
        from openpose import pyopenpose as op
    except ImportError:
        print('Warning: openpose has not been properly installed! Pre-stored pose will be used!')
        use_openpose_flag = False
        # ref_pose = np.load('input/ref_pose.npy')
        # test_poses = np.load('input/test_poses.npy')
    except Exception as e:
        raise e
    
    if use_openpose_flag:
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
    else:
        ref_pose = np.load('input/ref_pose.npy')
        test_poses = np.load('input/test_poses.npy')

    ref_pose = ref_pose[VIS_KP]
    ref_img_with_pose = display_pose(ref_img, ref_pose, color=(0, 255, 0))
    test_poses = []
    scores_list = []
    idx = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            if use_openpose_flag:
                # Process Test Image and Display
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])
                test_pose = datum.poseKeypoints[0]
            else:
                test_pose = test_poses[idx]
            idx = idx + 1
            
            test_pose = test_pose[VIS_KP]
            test_poses.append(test_pose)

            pd = PoseDiff()
            try:
                scores = pd.pose_diff(ref_pose, test_pose)
            except PoseException as e:
                print(e)
                scores = None
            scores_list.append(scores)
            test_img_with_pose = display_pose(
                test_img, test_pose, color=(0, 255, 255), scores=scores)
            out.write(test_img_with_pose)
        else:
            break

        return ref_pose, test_poses, ref_img_with_pose, restore_path, scores_list


if __name__ == '__main__':
    args = parse_args()

    ref_img = cv2.imread(args.ref)

    suffix = args.test.split('.')[-1]
    if suffix in {'jpg', 'jpeg', 'bmp', 'png'}:
        input_type = 'image'
    elif suffix in {'avi', 'mov', 'mp4'}:
        input_type = 'video'
    else:
        raise Exception('Unsupported input format!')

    if input_type == 'image':
        test_img = cv2.imread(args.test)

        ref_pose, test_pose, ref_img_with_pose, test_img_with_pose, scores = pose_diff_img(
            ref_img, test_img)

        if args.show:
            plt.subplot(1, 2, 1)
            plt.imshow(ref_img_with_pose[:, :, ::-1])
            plt.subplot(1, 2, 2)
            plt.imshow(test_img_with_pose[:, :, ::-1])
            plt.show()
        else:
            cv2.imwrite(os.path.join(args.result, 'ref.jpg'), ref_img_with_pose)
            cv2.imwrite(os.path.join(args.result, 'test.jpg'), test_img_with_pose)
    else:
        ref_pose, test_poses, ref_img_with_pose, restore_path, scores_list = pose_diff_video(
            ref_img, args.test, os.path.join(args.result, 'test.mp4'))

        cv2.imwrite(os.path.join(args.result, 'ref.jpg'), ref_img_with_pose)

        if args.show:
            print('not support for video input, please view the results in result/ directory')

