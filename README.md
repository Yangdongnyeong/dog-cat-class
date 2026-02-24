# Cat vs Dog Image Classifier 🐱🐶

This is a web-based artificial intelligence service that classifies images of cats and dogs. It runs a deep learning model (Xception) using **Flask** and provides a sleek, modern, dark-themed user interface for easy drag-and-drop image analysis.

## 🚀 Features
- **Drag & Drop Interface**: Easily upload images of cats or dogs.
- **Deep Learning Model**: Utilizes a pre-trained `Xception` model for accurate visual recognition.
- **Real-time Prediction**: Displays the classification result along with the model's confidence percentage.
- **Modern UI**: Dark-themed, glassmorphism design for a premium feel.

## 🛠️ Tech Stack
- **Backend**: Python, Flask, TensorFlow (Keras)
- **Frontend**: HTML, CSS (Vanilla), JavaScript
- **Image Processing**: Pillow, NumPy

## 📦 Requirements
The dependencies for this project are listed in `requirements.txt`:
- Flask
- Werkzeug
- tensorflow
- pillow

## ⚙️ Installation & Usage

1. **Clone the repository** (if applicable) or download the files.

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the Model**
   Ensure your trained keras model named `best_model_xception.keras` is placed in the root directory alongside `app.py`. *(Note: The model file is included in `.gitignore` and won't be pushed to the repository).*

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the Web Service**
   Open your browser and navigate to:
   👉 `http://localhost:8000`

## 📂 Project Structure
- `app.py`: The main Flask application containing the server logic, routing, and embedded HTML frontend.
- `requirements.txt`: Python package dependencies.
- `best_model_xception.keras`: The saved AI model (not included in version control).
- `.gitignore`: Specifies intentionally untracked files to ignore (e.g., the large `.keras` model).

## 💡 How it works
1. When an image is uploaded via the frontend, it is sent via a POST request to the `/predict` endpoint.
2. The image is converted and resized to `150x150` pixels to match the input shape expected by the model.
3. The image goes through `xception.preprocess_input` array scaling.
4. The TensorFlow model predicts the probability of the image being a Cat or a Dog.
5. The frontend dynamically updates the UI to show the winning class and a confidence bar.
