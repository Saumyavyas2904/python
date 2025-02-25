import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Stitching function: reads images, stitches them, then crops black borders.
def stitch_images(image_paths):
    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]
    if len(images) < 2:
        return None, "Need at least two images to stitch"
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        return None, f"Stitching failed! Error code: {status}"
    return crop_panorama(stitched), None

# Improved cropping function to remove all black borders
def crop_panorama(panorama):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return panorama
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)
    cropped = panorama[y:y+h, x:x+w]
    return cropped

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Stitching</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .upload-form { margin-bottom: 20px; }
    img { max-width: 100%; height: auto; }
    .error { color: red; }
  </style>
</head>
<body>
  <h1>Image Stitching</h1>
  {% if error %}
    <p class="error">Error: {{ error }}</p>
  {% endif %}
  {% if stitched_url %}
    <h2>Stitched Image</h2>
    <img src="{{ stitched_url }}" alt="Stitched Panorama">
    <br>
    <a href="/">Stitch More Images</a>
  {% else %}
    <form method="post" action="/" enctype="multipart/form-data" class="upload-form">
      <label>Select at least two images (png, jpg, jpeg, bmp):</label><br>
      <input type="file" name="images" multiple required><br><br>
      <button type="submit">Stitch Images</button>
    </form>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "images" not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No images uploaded", stitched_url=None)
        files = request.files.getlist("images")
        image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image_paths.append(filepath)
        if len(image_paths) < 2:
            return render_template_string(HTML_TEMPLATE, error="At least two images required", stitched_url=None)
        stitched_image, error = stitch_images(image_paths)
        if error:
            return render_template_string(HTML_TEMPLATE, error=error, stitched_url=None)
        
        # Generate a unique filename using timestamp
        unique_filename = f"stitched_{int(time.time())}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, unique_filename)
        cv2.imwrite(output_path, stitched_image)
        
        stitched_url = f"/stitched_image/{unique_filename}"
        return render_template_string(HTML_TEMPLATE, stitched_url=stitched_url, error=None)
    
    return render_template_string(HTML_TEMPLATE, stitched_url=None, error=None)

# Route to serve stitched images dynamically.
@app.route("/stitched_image/<filename>")
def get_stitched_image(filename):
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(output_path):
        return send_from_directory(OUTPUT_FOLDER, filename, mimetype="image/jpeg")
    else:
        return "No stitched image available", 404

if __name__ == "__main__":
    app.run(debug=True)
