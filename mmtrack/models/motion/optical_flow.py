from mmtrack.registry import TASK_UTILS
import numpy as np
import cv2
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from skimage.transform import SimilarityTransform


@TASK_UTILS.register_module()
class OpticalFlow:
    def __init__(self):
        self.startXs = None
        self.startYs = None

    def initiate(self, frame: np.ndarray):
        """
        Args:
            frame: (H, W, 3) array of frame
        """
        self.__old_frame = frame.copy()

    def track(self, bboxes: np.ndarray, frame: np.ndarray, use_shi: bool = False):
        """
        Args:
            bboxes: (N, 4) array of bboxes in format (x1, y1, x2, y2)
            frame: (H, W, 3) array of frame
        """
        if self.startXs is None:
            self.startXs, self.startYs = self.get_features(
                cv2.cvtColor(self.__old_frame, cv2.COLOR_RGB2GRAY), bboxes, use_shi=use_shi)

        newXs, newYs = self.estimate_translations(self.startXs, self.startYs, self.__old_frame, frame)

        bboxes = xyxy2corners(bboxes)
        Xs, Ys, new_bboxes = self.apply_transformation(self.startXs, self.startYs, newXs, newYs, bboxes)

        self.__old_frame = frame.copy()

        new_bboxes = corners2xyxy(new_bboxes)
        return new_bboxes, Xs, Ys

    def get_features(self, img: np.ndarray, bboxes: np.ndarray, use_shi: bool = True):
        """
        Args:
            bboxes: (N, 4) array of bboxes in format (x1, y1, x2, y2)
        """
        num_objs = bboxes.shape[0]
        N = 0
        temp = []

        clip_fn = lambda x, small, big: small if x < small else big if x > big else x
        for i in range(num_objs):
            (xmin, ymin, xmax, ymax) = bboxes[i].astype(np.int32)
            xmin = clip_fn(xmin, 0, img.shape[1] - 1)
            ymin = clip_fn(ymin, 0, img.shape[0] - 1)
            xmax = clip_fn(xmax, 0, img.shape[1] - 1)
            ymax = clip_fn(ymax, 0, img.shape[0] - 1)
            roi = img[ymin:ymax, xmin:xmax]

            if use_shi:
                corner_response = corner_shi_tomasi(roi)
            else:
                corner_response = corner_harris(roi)
            
            coors = peak_local_max(corner_response, num_peaks=20, exclude_border=2)
            coors[:, 1] += xmin
            coors[:, 0] += ymin
            temp.append(coors)

            if coors.shape[0] > N:
                N = coors.shape[0]
        x = np.full((N, num_objs), -1)
        y = np.full((N, num_objs), -1)

        for i in range(num_objs):
            num_feats = temp[i].shape[0]
            x[:num_feats, i] = temp[i][:, 1]
            y[:num_feats, i] = temp[i][:, 0]

        return x, y
    
    def estimate_features_translation(self, startX, startY, Ix, Iy, old_frame, new_frame):
        X = startX.copy()
        Y = startY.copy()
        WINDOW_SIZE = 25
        mesh_x, mesh_y = np.meshgrid(np.arange(WINDOW_SIZE), np.arange(WINDOW_SIZE))
        old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

        mesh_x_flat = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor_fix = np.vstack((mesh_x_flat, mesh_y_flat))

        I1_value = interp2(old_frame_gray, coor_fix[[0], :], coor_fix[[1], :])
        Ix_value = interp2(Ix, coor_fix[[0], :], coor_fix[[1], :])
        Iy_value = interp2(Iy, coor_fix[[0], :], coor_fix[[1], :])
        I = np.vstack((Ix_value, Iy_value))
        A = I.dot(I.T)

        for _ in range(15):
            mesh_x_flat = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
            mesh_y_flat = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
            coor = np.vstack((mesh_x_flat, mesh_y_flat))

            I2_value = interp2(new_frame_gray, coor[[0], :], coor[[1], :])
            Ip = (I2_value - I1_value).reshape((-1, 1))
            b = -I.dot(Ip)

            solution = np.linalg.inv(A).dot(b)
            X += solution[0, 0]
            Y += solution[1, 0]

        return X, Y
    
    def estimate_translations(self, startXs, startYs, old_frame: np.ndarray, new_frame: np.ndarray):
        """
        Args:
            starts: (N, 2) array of (x, y) coordinates
            old_frame: (H, W, 3) array of old frame
            new_frame: (H, W, 3) array of new frame
        """
        I = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
        I = cv2.GaussianBlur(I, (5, 5), 0.2)
        Iy, Ix = np.gradient(I.astype(np.float32))

        startXs_flat = startXs.flatten()
        startYs_flat = startYs.flatten()

        newXs = np.full_like(startXs_flat, -1, dtype=np.float32)
        newYs = np.full_like(startYs_flat, -1, dtype=np.float32)

        for i in range(len(startXs)):
            if startXs_flat[i] != -1:
                newXs[i], newYs[i] = self.estimate_features_translation(
                    startXs_flat[i], startYs_flat[i], Ix, Iy, old_frame, new_frame)
        
        newXs = newXs.reshape(startXs.shape)
        newYs = newYs.reshape(startYs.shape)
        return newXs, newYs

    def apply_transformation(self, startXs, startYs, newXs, newYs, bboxes: np.ndarray):
        """
        Args:
            starts: (N, 2) array of (x, y) coordinates
            ends: (N, 2) array of (x, y) coordinates
            bboxes: (N, 4) array of bboxes in format (x1, y1, x2, y2)
        """
        num_objs = bboxes.shape[0]
        new_bboxes = np.zeros_like(bboxes)

        Xs = newXs.copy()
        Ys = newYs.copy()

        for obj_ind in range(num_objs):
            startXs_obj = startXs[:, [obj_ind]]
            startYs_obj = startYs[:, [obj_ind]]
            newXs_obj = newXs[:, [obj_ind]]
            newYs_obj = newYs[:, [obj_ind]]

            desired_points = np.hstack((startXs_obj, startYs_obj))
            actual_points = np.hstack((newXs_obj, newYs_obj))

            transform = SimilarityTransform()
            transform.estimate(dst=actual_points, src=desired_points)
            mat = transform.params

            THRES = 80000
            projected = mat.dot(np.vstack((
                desired_points.T.astype(np.float32),
                np.ones((1, desired_points.shape[0])))))
            distance = np.square(projected[0:2, :].T - actual_points).sum(axis=1)

            actual_inliers = actual_points[distance < THRES]
            desired_inliers = desired_points[distance < THRES]

            if desired_inliers.shape[0] < 4:
                print('Two few points')
                actual_inliers = actual_points
                desired_inliers = desired_points
            transform.estimate(dst=actual_inliers, src=desired_inliers)
            mat = transform.params
            # coors = np.vstack((bboxes[obj_ind].T, np.array((1, 1, 1, 1))))
            coors = np.vstack((bboxes[obj_ind,:,:].T, np.array((1,1,1,1))))
            new_coors = mat.dot(coors)
            new_bboxes[obj_ind, :, :] = new_coors[0:2, :].T

            Xs[distance >= THRES, obj_ind] = -1
            Ys[distance >= THRES, obj_ind] = -1

        return Xs, Ys, new_bboxes


def xyxy2corners(boxes: np.ndarray):
    """
    (x1, y1, x2, y2) -> ((x1, y1), (x2, y1), (x1, y2), (x2, y2))

    Args:
        boxes: (N, 4) array of boxes in format (x1, y1, x2, y2)
    
    Returns:
        (N, 4, 2) array of boxes in format ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    new_boxes=  np.array([
        [x1, y1],
        [x2, y1],
        [x1, y2],
        [x2, y2],
    ])
    new_boxes = np.moveaxis(new_boxes, -1, 0)
    return new_boxes

def corners2xyxy(boxes: np.ndarray):
    """
    ((x1, y1), (x2, y1), (x1, y2), (x2, y2)) -> (x1, y1, x2, y2)

    Args:
        boxes: (N, 4, 2) array of boxes in format ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    
    Returns:
        (N, 4) array of boxes in format (x1, y1, x2, y2)
    """
    x1 = boxes[:, 0, 0]
    y1 = boxes[:, 0, 1]
    x2 = boxes[:, 3, 0]
    y2 = boxes[:, 3, 1]
    return np.array([x1, y1, x2, y2]).T

def interp2(v, xq, yq):

    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'


    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor<0] = 0
    y_floor[y_floor<0] = 0
    x_ceil[x_ceil<0] = 0
    y_ceil[y_ceil<0] = 0

    x_floor[x_floor>=w-1] = w-1
    y_floor[y_floor>=h-1] = h-1
    x_ceil[x_ceil>=w-1] = w-1
    y_ceil[y_ceil>=h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h,q_w)
    return interp_val
