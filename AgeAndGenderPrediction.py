# import cv2
# from deepface import DeepFace

# img = cv2.imread("faces/p4.jpeg")

# results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))

# print(results)





from flask import Flask, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is actually sent by the user
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read image file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
    
    # Perform analysis
    try:
        results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
