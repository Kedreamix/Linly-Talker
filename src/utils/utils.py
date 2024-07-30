import cv2
import numpy as np
from skimage import transform as trans


arcface_src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(lmk, face_size, dst_face_size, expand_size):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1) 
    min_M = []                                              
    min_index = []                                          
    min_error = float('inf')   

    assert face_size == 112
    src = (arcface_src / face_size * dst_face_size) + (expand_size - dst_face_size) / 2                
   
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def metrix_M(face_size, expand_size, keypoints=None):
    id_size = 112
    detected_lmk = np.concatenate(keypoints).reshape(5, 2)
    M, _ = estimate_norm(detected_lmk, id_size, face_size, expand_size)
    Minv = np.identity(3, dtype=np.single)
    Minv[0:2, :] = M
    M = Minv[0:2, :]
    return M   


def decompose_tfm(tfm):
    tfm = tfm.copy()
    s_x = np.sqrt(tfm[0][0] ** 2 + tfm[0][1] ** 2)
    s_y = np.sqrt(tfm[1][0] ** 2 + tfm[1][1] ** 2)

    t_x = tfm[0][2]
    t_y = tfm[1][2]

    #平移旋转矩阵rt
    rt = np.array([
        [tfm[0][0] / s_x, tfm[0][1] / s_x, t_x / s_x],
        [tfm[1][0] / s_y, tfm[1][1] / s_y, t_y / s_y],
    ])

    #缩放矩阵s
    s = np.array([
        [s_x, 0, 0],
        [0, s_y, 0]
    ])

    # _rt = np.vstack([rt, [[0, 0, 1]]])
    # _s = np.vstack([s, [[0, 0, 1]]])
    # print(np.dot(_s, _rt)[:2] - tfm)

    return rt, s


def img_warp(img, M, expand_size, adjust=0):
    warped = cv2.warpAffine(img, M, (expand_size, expand_size))
    warped = warped - np.uint8(adjust)
    warped = np.clip(warped, 0, 255)
    return warped


def img_warp_back_inv_m(img, img_to, inv_m):
    h_up, w_up, c = img_to.shape

    mask = np.ones_like(img).astype(np.float32)
    inv_mask = cv2.warpAffine(mask, inv_m, (w_up, h_up))
    inv_img = cv2.warpAffine(img, inv_m, (w_up, h_up))

    img_to[inv_mask == 1] = inv_img[inv_mask == 1]
    return img_to


def get_video_fps(vfile):
    cap = cv2.VideoCapture(vfile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


class laplacianSmooth(object):

    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
