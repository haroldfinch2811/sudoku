import cv2
import numpy as np
import os
import pytesseract

def extract_sudoku_grid(image_path):
    # Create debug directory if it doesn't exist
    debug_dir = "debug_output"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Clear previous debug images
    for file in os.listdir(debug_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(debug_dir, file))

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image")
        
    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Adaptive thresholding with adjusted parameters
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    
    # Invert image
    thresh = cv2.bitwise_not(thresh)
    
    # Dilate to fill gaps
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv2.dilate(thresh, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour with 4 corners (the puzzle grid)
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
        
    # Get perspective transform
    puzzle_contour = puzzle_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = puzzle_contour.sum(axis=1)
    rect[0] = puzzle_contour[np.argmin(s)]
    rect[2] = puzzle_contour[np.argmax(s)]
    
    diff = np.diff(puzzle_contour, axis=1)
    rect[1] = puzzle_contour[np.argmin(diff)]
    rect[3] = puzzle_contour[np.argmax(diff)]
    
    # Calculate dimensions
    width = max(
        int(np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))),
        int(np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)))
    )
    height = max(
        int(np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))),
        int(np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)))
    )
    
    # Create destination points
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (width, height))
    
    # Resize to larger fixed size for better digit recognition
    warped = cv2.resize(warped, (630, 630))  # Increased from 450x450
    
    # Split into 9x9 grid
    grid = []
    cell_height = warped.shape[0] // 9
    cell_width = warped.shape[1] // 9
    
    # Configure tesseract with adjusted parameters
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'  # Changed PSM mode
    
    for i in range(9):
        row = []
        for j in range(9):
            # Extract cell with larger margin
            margin = 8
            cell = warped[i*cell_height+margin:(i+1)*cell_height-margin, 
                         j*cell_width+margin:(j+1)*cell_width-margin]
            
            # Save original cell for debugging
            cv2.imwrite(f"{debug_dir}/cell_{i}_{j}_original.png", cell)
            
            # Calculate white pixel percentage for debugging
            # Note: In grayscale, white is 255 and black is 0
            # We consider pixels above 200 as "white" to account for slight variations
            white_pixels = np.sum(cell > 200)
            total_pixels = cell.size
            white_percentage = white_pixels / total_pixels
            
            # Save debugging info
            with open(f"{debug_dir}/cell_{i}_{j}_debug.txt", 'w') as f:
                f.write(f"White percentage: {white_percentage:.2f}\n")
            
            if white_percentage > 0.90:  # Adjusted threshold based on new calculation
                row.append(0)
            else:
                try:
                    digit = pytesseract.image_to_string(cell, config=custom_config).strip()
                    
                    # Save OCR result for debugging
                    with open(f"{debug_dir}/cell_{i}_{j}_ocr.txt", 'w') as f:
                        f.write(f"OCR result: {digit}\n")
                    
                    if digit and digit.isdigit() and 1 <= int(digit) <= 9:
                        row.append(int(digit))
                    else:
                        row.append(0)
                except Exception as e:
                    # Log any errors for debugging
                    with open(f"{debug_dir}/cell_{i}_{j}_error.txt", 'w') as f:
                        f.write(str(e))
                    row.append(0)
                
        grid.append(row)
        
    return grid

if __name__ == "__main__":
    # Example usage
    try:
        grid = extract_sudoku_grid("assets/puzzle_2.png")
        for row in grid:
            print(row)
    except Exception as e:
        print(f"Error: {str(e)}")
