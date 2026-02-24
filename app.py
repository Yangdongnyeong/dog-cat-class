import io
import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image

# Initialize Flask App
app = Flask(__name__)

# Load the Xception model
MODEL_PATH = "best_model_xception.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define Class Labels
CLASS_NAMES = ["Cat", "Dog"]

def preprocess_image(image_bytes: bytes):
    try:
        # Open image using PIL
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Xception target size is (150, 150)
        img = img.resize((150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Expand dimensions to match batch size (1, 150, 150, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Apply Xception specific preprocessing (scales pixels between -1 and 1)
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            # Read the file bytes
            contents = file.read()
            # Preprocess the image
            img_tensor = preprocess_image(contents)
            
            # Predict
            predictions = model.predict(img_tensor)
            
            # Parse prediction
            if predictions.shape[-1] == 1:
                # Sigmoid output
                confidence = float(predictions[0][0])
                predicted_class_idx = 1 if confidence > 0.5 else 0
                
                if predicted_class_idx == 0:
                    probability = 1.0 - confidence
                else:
                    probability = confidence
            else:
                # Softmax output
                predicted_class_idx = int(np.argmax(predictions[0]))
                probability = float(predictions[0][predicted_class_idx])
                
            return jsonify({
                "prediction": CLASS_NAMES[predicted_class_idx],
                "confidence": probability,
                "filename": file.filename
            })
        except Exception as e:
             return jsonify({"error": str(e)}), 500
             
    return jsonify({"error": "Only PNG, JPG, or JPEG images are supported."}), 400


@app.route("/", methods=["GET"])
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Cat vs Dog Detector</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text-color: #f8fafc;
                --text-muted: #94a3b8;
                --accent-cat: #ec4899;
                --accent-dog: #3b82f6;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background-image: 
                    radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.15) 0%, transparent 50%);
                overflow-x: hidden;
            }

            .container {
                width: 100%;
                max-width: 600px;
                padding: 2rem;
                position: relative;
            }

            .glass-panel {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 3rem 2rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .glass-panel:hover {
                transform: translateY(-5px);
                box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.6);
            }

            h1 {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            p.subtitle {
                color: var(--text-muted);
                margin-bottom: 2rem;
                font-size: 1.1rem;
            }

            .upload-area {
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 3rem 2rem;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                background: rgba(15, 23, 42, 0.4);
            }

            .upload-area:hover, .upload-area.dragover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.1);
            }

            .upload-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
                display: block;
                animation: float 3s ease-in-out infinite;
            }

            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0px); }
            }

            .upload-text {
                font-weight: 600;
                font-size: 1.2rem;
            }
            
            .upload-hint {
                color: var(--text-muted);
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }

            input[type="file"] {
                display: none;
            }

            .preview-container {
                display: none;
                margin-top: 1.5rem;
                position: relative;
            }

            .preview-container img {
                max-width: 100%;
                max-height: 300px;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
                object-fit: contain;
            }

            .btn {
                background: linear-gradient(135deg, var(--primary), #818cf8);
                color: white;
                border: none;
                padding: 1rem 2.5rem;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 2rem;
                box-shadow: 0 10px 20px -10px var(--primary);
                width: 100%;
                font-family: inherit;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 25px -10px var(--primary);
                background: linear-gradient(135deg, var(--primary-hover), var(--primary));
            }
            
            .btn:disabled {
                background: #475569;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            .result-container {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.6);
                display: none;
                opacity: 0;
                transform: translateY(20px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .result-container.show {
                display: block;
                opacity: 1;
                transform: translateY(0);
            }

            .result-title {
                font-size: 1.8rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }

            .result-cat { color: var(--accent-cat); }
            .result-dog { color: var(--accent-dog); }

            .confidence-bar {
                width: 100%;
                height: 8px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                margin-top: 1rem;
                overflow: hidden;
                position: relative;
            }

            .confidence-fill {
                height: 100%;
                border-radius: 4px;
                width: 0%;
                transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .fill-cat { background: linear-gradient(90deg, #f472b6, #db2777); }
            .fill-dog { background: linear-gradient(90deg, #60a5fa, #2563eb); }

            .confidence-text {
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: var(--text-muted);
                display: flex;
                justify-content: space-between;
            }

            /* Loader */
            .loader {
                display: none;
                width: 48px;
                height: 48px;
                border: 4px solid rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                border-top-color: var(--primary);
                animation: spin 1s ease-in-out infinite;
                margin: 2rem auto;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .error-message {
                color: #ef4444;
                margin-top: 1rem;
                font-size: 0.9rem;
                display: none;
                background: rgba(239, 68, 68, 0.1);
                padding: 0.75rem;
                border-radius: 8px;
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="glass-panel">
                <h1>AI Vision</h1>
                <p class="subtitle">Cat vs Dog Neural Core</p>

                <div class="upload-area" id="drop-zone">
                    <span class="upload-icon">📸</span>
                    <div class="upload-text">Drag & drop an image here</div>
                    <div class="upload-hint">or click to browse</div>
                    <input type="file" id="file-input" accept="image/jpeg, image/png, image/jpg">
                </div>

                <div class="preview-container" id="preview-container">
                    <img id="image-preview" src="" alt="Preview">
                </div>
                
                <div class="error-message" id="error-message"></div>

                <button class="btn" id="predict-btn" disabled>Analyze Image</button>

                <div class="loader" id="loader"></div>

                <div class="result-container" id="result-container">
                    <div class="result-title" id="result-text">---</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-fill"></div>
                    </div>
                    <div class="confidence-text">
                        <span>Confidence</span>
                        <span id="confidence-val">0%</span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('drop-zone');
            const fileInput = document.getElementById('file-input');
            const previewContainer = document.getElementById('preview-container');
            const imagePreview = document.getElementById('image-preview');
            const predictBtn = document.getElementById('predict-btn');
            const loader = document.getElementById('loader');
            const resultContainer = document.getElementById('result-container');
            const resultText = document.getElementById('result-text');
            const confidenceFill = document.getElementById('confidence-fill');
            const confidenceVal = document.getElementById('confidence-val');
            const errorMessage = document.getElementById('error-message');

            let selectedFile = null;

            // Handle clicking on drop zone
            dropZone.addEventListener('click', () => fileInput.click());

            // Handle drag events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
            });

            // Handle file drop
            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                handleFiles(files);
            });

            // Handle file selection
            fileInput.addEventListener('change', function() {
                handleFiles(this.files);
            });

            function handleFiles(files) {
                if (files.length === 0) return;
                
                const file = files[0];
                if (!file.type.match('image.*')) {
                    showError("Please upload an image file (JPG, PNG).");
                    return;
                }

                selectedFile = file;
                hideError();
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    dropZone.style.display = 'none';
                    previewContainer.style.display = 'block';
                    predictBtn.disabled = false;
                    
                    // Reset result
                    resultContainer.classList.remove('show');
                    setTimeout(() => {
                        resultContainer.style.display = 'none';
                        confidenceFill.style.width = '0%';
                    }, 300);
                }
                reader.readAsDataURL(file);
            }

            function showError(msg) {
                errorMessage.textContent = msg;
                errorMessage.style.display = 'block';
            }
            
            function hideError() {
                errorMessage.style.display = 'none';
            }

            // Allow clicking preview to change image
            imagePreview.addEventListener('click', () => {
                fileInput.click();
            });

            // Predict
            predictBtn.addEventListener('click', async () => {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('file', selectedFile);

                // UI Loading state
                predictBtn.style.display = 'none';
                loader.style.display = 'block';
                resultContainer.classList.remove('show');
                resultContainer.style.display = 'none';
                hideError();

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.error || "Prediction failed.");
                    }

                    // Show results
                    const isCat = data.prediction.toLowerCase() === 'cat';
                    const confPercent = (data.confidence * 100).toFixed(1);
                    
                    resultText.textContent = data.prediction;
                    resultText.className = 'result-title ' + (isCat ? 'result-cat' : 'result-dog');
                    
                    confidenceFill.className = 'confidence-fill ' + (isCat ? 'fill-cat' : 'fill-dog');
                    confidenceVal.textContent = confPercent + '%';
                    
                    resultContainer.style.display = 'block';
                    // Trigger reflow
                    void resultContainer.offsetWidth;
                    resultContainer.classList.add('show');
                    
                    // Animate bar
                    setTimeout(() => {
                        confidenceFill.style.width = confPercent + '%';
                    }, 100);

                } catch (err) {
                    showError(err.message);
                } finally {
                    loader.style.display = 'none';
                    predictBtn.style.display = 'block';
                    predictBtn.textContent = 'Analyze Another Image';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
