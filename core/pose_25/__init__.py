VIS_KP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
JOINT = ['nose',
         'neck',
         'right shoulder',
         'right elbow',
         'right wrist',
         'left shoulder',
         'left elbow',
         'left wrist',
         'waist',
         'right hip',
         'right knee',
         'right ankle',
         'left hip',
         'left knee',
         'left ankle']
LIMBS = [[1, 0], [8, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
         [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

KP_TREE = {}
KP_TREE[1] = [5, 2, 0]
KP_TREE[2] = [3]
KP_TREE[3] = [4]
# KP_TREE[4] = []
KP_TREE[5] = [6]
KP_TREE[6] = [7]
# KP_TREE[7] = []
KP_TREE[8] = [12, 9, 1]
KP_TREE[9] = [10]
KP_TREE[10] = [11]
# KP_TREE[11] = []
KP_TREE[12] = [13]
KP_TREE[13] = [14]
# KP_TREE[14] = []
