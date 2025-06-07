import cv2
import numpy as np
from io import BytesIO
import requests
from flask import Flask, request, send_file, render_template, redirect, url_for
import os

app = Flask(__name__)

# Function to filter out placeholder pieces and retain only real ones
def filter_real_pieces_with_inpainting(image):

    # API URL for chessboard detection
    api_url = "https://helpman.komtera.lt/chessocr/predict"
 
    # Convert the image (from NumPy array to file-like object) to pass to the API
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # Prepare the image for the API request
    files = {"file": ("chessboard.jpg", img_bytes, "image/jpg")}
    response = requests.post(api_url, files=files)

    if response.status_code != 200:
        print("Error: API call failed with status code", response.status_code)
        print("Response content:", response.content)  # Print response content
        exit(1)

    data = response.json()
    if "results" not in data or not data["results"]:
        print("Error: No results from API")
        exit(1)

    # Extract xc, yc, width, height from the first result in the API response
    result = data["results"][0]
    xc = result["xc"]
    yc = result["yc"]
    board_width = result["width"]
    board_height = result["height"]
    output_size = (500, 500)
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Convert normalized values to actual pixel dimensions
    actual_width = board_width * img_width
    actual_height = board_height * img_height
    center_x = xc * img_width
    center_y = yc * img_height

    # Calculate bounding box coordinates
    x1 = int(center_x - actual_width / 2)
    y1 = int(center_y - actual_height / 2)
    x2 = int(center_x + actual_width / 2)
    y2 = int(center_y + actual_height / 2)

    # Ensure coordinates are within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    cropped_image = image[y1:y2, x1:x2]
    image_resized = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

    # Adjust the image to be divisible by 8 (for chessboard grid)
    height, width = image_resized.shape[:2]
    new_width = int(round(width / 8.0)) * 8
    new_height = int(round(height / 8.0)) * 8
    if new_width != width or new_height != height:
        image_resized = cv2.resize(image_resized, (new_width, new_height))

    # Step 2: Define chessboard grid size (after resizing)
    height, width = image_resized.shape[:2]
    square_size_x = width // 8
    square_size_y = height // 8

    filtered_image = image_resized.copy()

    # Loop through each square and check for real pieces or placeholders
    for row in range(8):
        for col in range(8):
            x_start = col * square_size_x
            y_start = row * square_size_y
            crop_size = int(square_size_x * 0.2)
            square = filtered_image[y_start + crop_size:y_start + square_size_y - crop_size,
                                    x_start + crop_size:x_start + square_size_x - crop_size]

            gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            # edges = cv2.Canny(gray_square, threshold1=100, threshold2=200)
            variance = np.var(gray_square)
            # print(f"Variance for square ({row}, {col}): {variance}")
            #printing new line
            # print("\n")
            

            if variance < 300:
                surrounding_color = np.mean(filtered_image[y_start:y_start + square_size_y,
                                                           x_start:x_start + square_size_x], axis=(0, 1))
                surrounding_color = tuple(int(c) for c in surrounding_color)

                # Fill the fake region with the surrounding color
                filtered_image[y_start :y_start + square_size_y,
                               x_start :x_start + square_size_x ] = surrounding_color
    
    
    return filtered_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    f = request.files['image']
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    filtered_image = filter_real_pieces_with_inpainting(img)
    processed_save_path = os.path.join(app.static_folder, 'processed_image.jpg')
    real_save_path = os.path.join(app.static_folder, 'real_image.jpg')
    cv2.imwrite(real_save_path, img)
    cv2.imwrite(processed_save_path, filtered_image)

    url = "https://helpman.komtera.lt/chessocr/predict"

    with open(processed_save_path, "rb") as image_file:
        files = {"file": ("image.jpg", image_file, "image/jpg")}
        response = requests.post(url, files=files)
        data = response.json()
   
    fen = data["results"][0]["fen"]
    print(f"FEN: {fen}")
    # default move = black, default orientation = white
    return redirect(url_for('analysis', fen=fen, move='b', orientation='white'))

@app.route('/analysis')
def analysis():
    fen         = request.args['fen']
    move        = request.args.get('move','b')        # 'b' or 'w'
    orientation = request.args.get('orientation','white')  # 'white' or 'black'
    return render_template('analysis.html',
                           fen=fen,
                           move=move,
                           orientation=orientation)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # listen on all interfaces so Railway / Docker / Vercel can route traffic
    app.run(host="0.0.0.0", port=port)
