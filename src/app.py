from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)
app.template_folder = "templates"

def is_valid(board, row, col, num):
    for x in range(9):
        if board[row][x] == num and x != col:
            return False
    
    for x in range(9):
        if board[x][col] == num and x != row:
            return False
    
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                if (i + start_row != row or j + start_col != col):
                    return False
    
    return True

def find_empty(board):
    min_possibilities = 10
    best_cell = None
    
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                count = sum(1 for num in range(1, 10) if is_valid(board, i, j, num))
                if count < min_possibilities:
                    min_possibilities = count
                    best_cell = (i, j)
                    if count == 1:
                        return best_cell
    
    return best_cell

def solve(board, depth=0):
    if depth > 100:
        return False
        
    empty = find_empty(board)
    if not empty:
        return True
        
    row, col = empty
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            
            if solve(board, depth + 1):
                return True
                
            board[row][col] = 0
            
    return False

def extract_puzzle_from_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    thresh = cv2.bitwise_not(thresh)
    
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv2.dilate(thresh, kernel)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    puzzle_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4 and area > max_area:
            puzzle_contour = approx
            max_area = area
            
    if puzzle_contour is None:
        raise Exception("Could not find Sudoku grid")
        
    puzzle_contour = puzzle_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = puzzle_contour.sum(axis=1)
    rect[0] = puzzle_contour[np.argmin(s)]
    rect[2] = puzzle_contour[np.argmax(s)]
    
    diff = np.diff(puzzle_contour, axis=1)
    rect[1] = puzzle_contour[np.argmin(diff)]
    rect[3] = puzzle_contour[np.argmax(diff)]
    
    width = max(
        int(np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))),
        int(np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)))
    )
    height = max(
        int(np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))),
        int(np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)))
    )
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (width, height))
    warped = cv2.resize(warped, (630, 630))
    
    grid = []
    cell_height = warped.shape[0] // 9
    cell_width = warped.shape[1] // 9
    
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    
    for i in range(9):
        row = []
        for j in range(9):
            margin = 8
            cell = warped[i*cell_height+margin:(i+1)*cell_height-margin, 
                         j*cell_width+margin:(j+1)*cell_width-margin]
            
            white_pixels = np.sum(cell > 200)
            total_pixels = cell.size
            white_percentage = white_pixels / total_pixels
            
            if white_percentage > 0.95:
                row.append(0)
            else:
                try:
                    digit = pytesseract.image_to_string(cell, config=custom_config).strip()
                    if digit and digit.isdigit() and 1 <= int(digit) <= 9:
                        row.append(int(digit))
                    else:
                        row.append(0)
                except:
                    row.append(0)
                
        grid.append(row)
        
    return grid

@app.route("/solve", methods=["POST"])
def solve_sudoku():
    data = request.get_json()
    puzzle = data.get("puzzle")
    
    if not puzzle or not isinstance(puzzle, list) or len(puzzle) != 9:
        return jsonify({"error": "Invalid puzzle format. Expected 9x9 grid"}), 400
        
    for row in puzzle:
        if not isinstance(row, list) or len(row) != 9:
            return jsonify({"error": "Invalid puzzle format. Expected 9x9 grid"}), 400
        for num in row:
            if not isinstance(num, int) or num < 0 or num > 9:
                return jsonify({"error": "Invalid numbers. Use 0-9 only"}), 400

    solution = [row[:] for row in puzzle]
    
    if solve(solution):
        return jsonify({
            "solution": solution
        })
    else:
        return jsonify({
            "error": "No solution exists for this puzzle"
        }), 400

@app.route("/", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded")
            
        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        image_data = file.read()
        try:
            puzzle = extract_puzzle_from_image(image_data)
            solution = [row[:] for row in puzzle]
            
            if solve(solution):
                return render_template("upload.html", puzzle=puzzle, solution=solution)
            else:
                return render_template("upload.html", puzzle=puzzle, error="Could not solve puzzle")
                
        except Exception as e:
            return render_template("upload.html", error=f"Error processing image: {str(e)}")

    return render_template("upload.html")
