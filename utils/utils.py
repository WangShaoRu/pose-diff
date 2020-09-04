import cv2
import numpy as np
from core.pose_25 import VIS_KP, LIMBS, KP_TREE


def display_pose(image, pose, vis_thr=0.6, color=(0, 255, 255), scores=None):
    def _check(kp, h, w):
        return True if 0 < kp[0] < w and 0 < kp[1] < h else False

    canvas = image.copy()
    (h, w, _) = canvas.shape
    for i, kp in enumerate(pose):
        if _check(kp, h, w):
            if scores is not None:
                score = scores[i]
                color = cv2.applyColorMap(
                    np.uint8(score * 255).reshape(1, 1), cv2.COLORMAP_AUTUMN).squeeze().tolist()
            if kp[2] > vis_thr:
                cv2.circle(canvas, (kp[0], kp[1]), 4, color, -1)
            else:
                cv2.circle(canvas, (kp[0], kp[1]), 4, color, 1)
    for pair in LIMBS:
        kp1, kp2 = pose[pair[0]], pose[pair[1]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if _check(kp1, h, w) and _check(kp2, h, w):
            if scores is not None:
                score = scores[pair[1]]
                color = cv2.applyColorMap(
                    np.uint8(score * 255).reshape(1, 1), cv2.COLORMAP_AUTUMN).squeeze().tolist()
                cv2.line(canvas, tuple(kp1[:2]), tuple(
                    kp2[:2]), color, thickness=1)
                cv2.putText(canvas, "%.2f" % score, tuple(
                    (kp1[:2]+kp2[:2])/2), font, 0.3, color)
            else:
                cv2.line(canvas, tuple(kp1[:2]), tuple(
                    kp2[:2]), color, thickness=1)

    return canvas