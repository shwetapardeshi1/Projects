import numpy as np
import cv2
import operator
from matplotlib import pyplot as plt
from keras.models import model_from_json


with open('models/model.json', 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model.h5")


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	
	img = inp_img.copy()  
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			
			if img.item(y, x) == 255 and x < width and y < height: 
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  
					max_area = area[0]
					seed_point = (x, y)

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8) 

	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64: 
				cv2.floodFill(img, mask, (x, y), 0)

			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point

def extract_digit(img, rect, size):

	digit = cut_from_rect(img, rect)  

	h, w = digit.shape[:2]
	margin = int(np.mean([h, w]) / 2.5)
	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_from_rect(digit, bbox)

	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

def show_digits(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
        img = np.concatenate(rows)
    return img


def cut_from_rect(img, rect):
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
	
def get_digits(img, squares, size):
 
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits

def parse_grid(path):
	original = cv2.imread(path)
	proc = cv2.GaussianBlur(original .copy(), (9, 9), 0)
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	proc = cv2.bitwise_not(proc, proc)

	if not skip_dilate:
		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		proc = cv2.dilate(proc, kernel)
    
	contours, h = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	polygon = contours[0]  
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    
	top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    
	dis1 = np.sqrt(((bottom_right[0] -top_right[0]) ** 2) + ((bottom_right[0] - top_right[0]) ** 2))
	dis2 = np.sqrt(((bottom_left[0] -top_left[0]) ** 2) + ((bottom_left[0] - top_left[0]) ** 2))
	dis3 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[0] -  bottom_left[0]) ** 2))
	dis4 = np.sqrt(((top_left[0] -top_right[0]) ** 2) + ((top_left[0] - top_right[0]) ** 2))
    
	side = max([ dis1, dis2, dis3, dis4])

	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	m = cv2.getPerspectiveTransform(src, dst)
	cropped = cv2.warpPerspective(img, m, (int(side), int(side)))

	squares = []
	side = cropped.shape[:1]
	side = side[0] / 9

	
	for j in range(9):
		for i in range(9):
			p1 = (i * side, j * side)  
			p2 = ((i + 1) * side, (j + 1) * side)  
			squares.append((p1, p2))
            
	digits = get_digits(cropped, squares, 28)
	final_image = show_digits(digits)
	return final_image

def extract_sudoku(image_path):
    final_image = parse_grid(image_path)
    return final_image


def identify_number(image):
    image_resize = cv2.resize(image, (28,28))  
    image_resize_2 = image_resize.reshape(1,1,28,28)    
    loaded_model_pred = loaded_model.predict_classes(image_resize_2 , verbose = 0)
    return loaded_model_pred[0]


def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
            if image.sum() > 25000:
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)


def main(image_path):
    image = extract_sudoku(image_path)
    grid = extract_number(image)
    print(grid)
    return grid


SIZE = 9
matrix = main("image1.jpg")

def print_sudoku():
    for i in matrix:
        print (i)

def number_unassigned(row, col):
    num_unassign = 0
    for i in range(0,SIZE):
        for j in range (0,SIZE):
            if matrix[i][j] == 0:
                row = i
                col = j
                num_unassign = 1
                a = [row, col, num_unassign]
                return a
    a = [-1, -1, num_unassign]
    return a

def is_safe(n, r, c):
    for i in range(0,SIZE):
        if matrix[r][i] == n:
            return False
    for i in range(0,SIZE):
        if matrix[i][c] == n:
            return False
    row_start = (r//3)*3
    col_start = (c//3)*3;
 
    for i in range(row_start,row_start+3):
        for j in range(col_start,col_start+3):
            if matrix[i][j]==n:
                return False
    return True

def solve_sudoku():
    row = 0
    col = 0

    a = number_unassigned(row, col)
    if a[2] == 0:
        return True
    row = a[0]
    col = a[1]
    for i in range(1,10):
        if is_safe(i, row, col):
            matrix[row][col] = i
            if solve_sudoku():
                return True
    
            matrix[row][col]=0
    return False

if solve_sudoku():
    print_sudoku()




