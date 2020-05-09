from utils import *
from keras.models import model_from_json


with open('model.json', 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")



def parse(path):
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




