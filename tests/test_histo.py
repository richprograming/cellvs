import cv2
import numpy as np

from cellvs.histo import crop_circle, through_eyepiece


def test_eyepiece_detection():
    eye = np.zeros((800, 600, 3), np.uint8)
    cv2.circle(eye, (300, 400), 250, (180, 120, 230), -1)
    scan = np.full((800, 600, 3), (200, 170, 230), np.uint8)
    assert through_eyepiece(eye) and not through_eyepiece(scan)


def test_crop_circle_stays_inside_the_eyepiece():
    img = np.zeros((800, 600, 3), np.uint8)
    cv2.circle(img, (320, 420), 200, (180, 120, 230), -1)
    crop = crop_circle(img)
    assert abs(crop.shape[0] - crop.shape[1]) <= 1
    assert 250 < crop.shape[0] < 2 * 200 / np.sqrt(2)
    assert crop.min() > 0  # no black border from outside the circle
