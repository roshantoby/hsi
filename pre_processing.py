import numpy as np
import cv2


def get_roi_pixels(img, cnt, dim=(256, 320)):
    ctr = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
    img_copy = np.zeros(dim)
    im_reshape = img.reshape(*dim, -1)
    cv2.drawContours(img_copy, [ctr], contourIdx=-1, color=100, thickness=cv2.FILLED)
    uv = np.where(img_copy == 100)
    return im_reshape[uv[0], uv[1], :]


turkey = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/turkey_1_HYSPEC.csv", delimiter='\t', skip_header=True)
chicken = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/chicken_1_HYSPEC.csv", delimiter='\t',  skip_header=True)
ham = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/ham_1_HYSPEC.csv", delimiter='\t', skip_header=True)

turkey_roi = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/turkey_roi.csv", delimiter=",", skip_header=True)
chicken_roi = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/chicken_roi.csv", delimiter=",", skip_header=True)
ham_roi = np.genfromtxt("C:/Users/rosha/OneDrive/Desktop/train/ham_roi.csv", delimiter=",",  skip_header=True)


turkey_op = get_roi_pixels(turkey, turkey_roi, dim=(256, 320))
chicken_op = get_roi_pixels(chicken, chicken_roi, dim=(256, 320))
ham_op = get_roi_pixels(ham, ham_roi, dim=(256, 320))

turkey_samples = np.c_[turkey_op, np.repeat(0, len(turkey_op))]
chicken_samples = np.c_[chicken_op, np.repeat(1, len(chicken_op))]
ham_samples = np.c_[ham_op, np.repeat(2, len(ham_op))]

samples = np.r_[turkey_samples, chicken_samples, ham_samples]

np.save('data/samples.npy', samples)