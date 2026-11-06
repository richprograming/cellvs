import cv2
import numpy as np

from cellvs.histo import crop_circle


def test_crop_circle_stays_inside_the_eyepiece():
    img = np.zeros((800, 600, 3), np.uint8)
    cv2.circle(img, (320, 420), 200, (180, 120, 230), -1)
    crop = crop_circle(img)
    assert abs(crop.shape[0] - crop.shape[1]) <= 1
    assert 250 < crop.shape[0] < 2 * 200 / np.sqrt(2)
    assert crop.min() > 0  # no black border from outside the circle
