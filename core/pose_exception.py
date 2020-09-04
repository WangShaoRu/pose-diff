class BodyTooShortException(Exception):
    "This exception happens when the body is too short to align"
    def __init__(self, name, len):
        self.name = name
        self.len = len

    def __str__(self):
        print("Body of %s is too short (%d)" % (self.name, self.len))