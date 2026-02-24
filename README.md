# Cat vs Dog Image Classifier 🐱🐶

이 프로젝트는 고양이(Cat)와 강아지(Dog)의 이미지를 분류하는 웹 기반 인공지능(AI) 서비스입니다. **Flask**를 사용하여 딥러닝 모델(Xception)을 실행하며, 사용자가 이미지를 쉽게 업로드(Drag & Drop)할 수 있도록 세련되고 모던한 다크 테마(Dark Theme) UI를 제공합니다.

## 🚀 주요 기능 (Features)
- **Drag & Drop 인터페이스**: 고양이 또는 강아지 이미지를 간편하게 업로드할 수 있습니다.
- **Deep Learning Model**: 영상 인식에 뛰어난 사전 학습된 `Xception` 모델을 활용하여 정확한 분석을 제공합니다.
- **Real-time Prediction**: 모델의 예측 확률(Confidence)과 함께 실시간으로 분류 결과를 화면에 표시합니다.
- **Modern UI**: 글래스모피즘(Glassmorphism)과 다크 테마를 적용하여 고급스러운 시각 디자인을 구현했습니다.

## 🛠️ 기술 스택 (Tech Stack)
- **Backend**: Python, Flask, TensorFlow (Keras)
- **Frontend**: HTML, CSS (Vanilla), JavaScript
- **Image Processing**: Pillow, NumPy

## 📦 필요 패키지 (Requirements)
이 프로젝트를 실행하기 위해 필요한 패키지는 `requirements.txt`에 명시되어 있습니다:
- Flask
- Werkzeug
- tensorflow
- pillow

## ⚙️ 설치 및 실행 방법 (Installation & Usage)

1. **Repository 다운로드** (또는 파일 복사)

2. **종속성 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **모델 파일 준비**
   사전 학습된 Keras 모델 파일인 `best_model_xception.keras`를 `app.py`와 같은 루트 디렉토리에 위치시킵니다. *(참고: 모델 파일은 용량이 크므로 `.gitignore` 파일에 의해 깃허브 버전 관리에 포함되지 않습니다).*

4. **웹 애플리케이션 실행**
   ```bash
   python app.py
   ```

5. **웹 서비스 접속**
   터미널에서 서버가 성공적으로 실행되면, 브라우저를 열고 다음 주소로 이동합니다:
   👉 `http://localhost:8000`

## 📂 프로젝트 구조 (Project Structure)
- `app.py`: 라우팅(Routing), 서버 로직, 프론트엔드 HTML이 포함된 메인 Flask 애플리케이션 파일입니다.
- `requirements.txt`: Python 패키지 종속성 정의 파일입니다.
- `best_model_xception.keras`: 저장된 AI 모델 파일 (버전 관리 제외).
- `.gitignore`: 깃허브에 업로드하지 않을 파일 목록(예: 모델 파일)을 정의한 파일입니다.

## 💡 분석 원리 (How it works)
1. 프론트엔드를 통해 이미지가 업로드되면, `/predict` 엔드포인트(Endpoint)로 `POST` 요청(Request)이 전달됩니다.
2. 모델이 요구하는 입력 크기(Input Shape)에 맞게 이미지를 `150x150` 픽셀로 변환 및 리사이징(Resizing)합니다.
3. 변환된 이미지는 `xception.preprocess_input`을 통해 모델에 맞게 배열 스케일링(Array Scaling) 처리됩니다.
4. TensorFlow 모델이 해당 이미지가 강아지(Dog)인지 고양이(Cat)인지 판단하고 확률(Probability)을 예측합니다.
5. 프론트엔드는 이 예측 데이터를 바탕으로 도출된 결과(Winning Class)와 동적으로 채워지는 확률 바(Confidence Bar)를 UI에 업데이트(Rendering)합니다.
