from utils import *
from keras.models import model_from_json


with open('model.json', 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")



def parse(path):
	image = cv2.imread(path)
	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)
	# find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	largestContourArea = 0
	largestContour = 0
	for cnt in cnts:
	contourArea = cv2.contourArea(cnt)
	if( contourArea > largestContourArea):
	    largestContour = cnt
	    largestContourArea = contourArea

	x,y,w,h = cv2.boundingRect(largestContour)

	cropped= image[y:y+h,x:x+w]
	cv2.namedWindow("Largest Contour",cv2.WINDOW_NORMAL)
	cv2.imshow("Largest Contour",ROI)
	cv2.waitKey(0)
	squares = []
	side = cropped.shape[:1]
	side = side[0] / 9
	for j in range(9):
	for i in range(9):
	    p1 = (i * side, j * side)  # Top left corner of a bounding box
	    p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
	    squares.append((p1, p2))

	digits = []

	cropped= imutils.resize(cropped, height = 300)
	gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	size = 28 
	for square in squares:
	digits.append(extract_digit(edged, square, size))

	final_image = show_digits(digits)
	return final_image


def extract_sudoku(image_path):
    final_image = (image_path)
    return final_image


def identify_number(image):
    image_resize = cv2.resize(image, (28,28))  
    image_resize_2 = image_resize.reshape(1,1,28,28)    
    loaded_model_pred = loaded_model.predict_classes(image_resize_2 , verbose = 0)
    return loaded_model_pred[0]


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




