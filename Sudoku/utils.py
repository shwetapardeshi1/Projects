import numpy as np
import cv2
import operator
from matplotlib import pyplot as plt\

def show_digits(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
        img = np.concatenate(rows)
    return img

def centre_pad(length):
 
    if length % 2 == 0:
        side1 = int((size - length) / 2)
        side2 = side1
    else:
        side1 = int((size - length) / 2)
        side2 = side1 + 1
    return side1, side2

def scale(r, x):
    return int(r * x)
    
def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]
    if h > w:
      t_pad = int(margin / 2)
      b_pad = t_pad
      ratio = (size - margin) / h
      w, h = scale(ratio, w), scale(ratio, h)
      l_pad, r_pad = centre_pad(w)
    else:
      l_pad = int(margin / 2)
      r_pad = l_pad
      ratio = (size - margin) / w
      w, h = scale(ratio, w), scale(ratio, h)
      t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def extract_digit(img, rect, size):


    digit = img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])] 

    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = digit[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]


    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
      return scale_and_centre(digit, size, 4)
    else:
      return np.zeros((size, size), np.uint8)
      
 def get_digits(img, squares, size):
    
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)

    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits

  
