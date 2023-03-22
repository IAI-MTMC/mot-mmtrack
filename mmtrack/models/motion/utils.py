import numpy as np

class Track:
    def __init__(self, tlbr, score, track_id):
        self.tlbr = tlbr
        self.score = score
        self.track_id = track_id
        self.inlier_ratio = 1.

        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)
    
    def __lt__(self, other):
        return self.tlbr[-1] < other.tlbr[-1]

    def set_keypoints(self, keypoints):
        self.prev_keypoints = self.keypoints
        self.keypoints = keypoints

def get_size(tlbr):
    return tlbr[2] - tlbr[0] + 1, tlbr[3] - tlbr[1] + 1

def aspect_ratio(tlbr):
    w, h = get_size(tlbr)
    return h / w if w > 0 else 0.

def get_center(tlbr):
    return (tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2

def to_tlbr(tlwh):
    tlbr = np.empty(4)
    xmin = float(tlwh[0])
    ymin = float(tlwh[1])
    tlbr[0] = round(xmin, 0)
    tlbr[1] = round(ymin, 0)
    tlbr[2] = round(xmin + float(tlwh[2]) - 1., 0)
    tlbr[3] = round(ymin + float(tlwh[3]) - 1., 0)
    return tlbr

def intersection(tlbr1, tlbr2):
    tlbr = np.empty(4)
    tlbr[0] = max(tlbr1[0], tlbr2[0])
    tlbr[1] = max(tlbr1[1], tlbr2[1])
    tlbr[2] = min(tlbr1[2], tlbr2[2])
    tlbr[3] = min(tlbr1[3], tlbr2[3])
    if tlbr[2] < tlbr[0] or tlbr[3] < tlbr[1]:
        return None
    return tlbr

def crop(img, tlbr):
    assert tlbr is not None
    xmin = max(int(tlbr[0]), 0)
    ymin = max(int(tlbr[1]), 0)
    xmax = max(int(tlbr[2]), 0)
    ymax = max(int(tlbr[3]), 0)
    return img[ymin:ymax + 1, xmin:xmax + 1]

def mask_area(mask):
    """Utility to calculate the area of a mask."""
    count = 0
    for val in mask.ravel():
        if val != 0:
            count += 1
    return count

def transform(pts, m):
    """Numpy implementation of OpenCV's transform."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.atleast_2d(pts)

    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T
