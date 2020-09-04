from .pose_25 import JOINT


class PoseException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        pass

class BodyTooShortException(PoseException):
    "This exception happens when the body is too short to align"
    def __init__(self, name, len):
        self.name = name
        self.len = len

    def __str__(self):
        print("Body of %s is too short (%d)" % (self.name, self.len))

class KPMissingException(PoseException):
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx
        self.joint = JOINT[idx]

    def __str__(self):
        print("%s of %s is missing" % (self.joint, self.name))