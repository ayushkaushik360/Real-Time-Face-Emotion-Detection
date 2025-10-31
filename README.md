



# 🎭 Real-Time Face Emotion Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **real-time facial emotion detection system** using **Convolutional Neural Networks (CNNs)** trained on the **FER2013 dataset**.
The system detects human emotions such as *happy, sad, angry, surprised,* and *neutral* directly from live webcam video feeds.

---

## 🧠 Features

✅ **Deep Learning Model:** Multi-layer CNN trained on the FER2013 dataset.<\n>
✅ **Real-Time Detection:** Face detection and emotion classification via **OpenCV** with **15–20 FPS** performance.<\n>
✅ **Regularization:** Dropout layers to reduce overfitting and improve generalization.<\n>
✅ **Lighting Optimization:** Adjusted preprocessing pipeline for variable lighting conditions.<\n>
✅ **Evaluation Metrics:** Includes accuracy, confusion matrix, and visualizations using Matplotlib.<\n>
✅ **Accuracy:** Achieved ~72% accuracy during real-time testing.</n>

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:**

  * TensorFlow / Keras
  * OpenCV
  * NumPy
  * Matplotlib

---

## 📂 Dataset

**FER2013** (Facial Expression Recognition 2013)

* Source: [Kaggle - FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* Classes:
  `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Real-Time-Face-Emotion-Detection.git
cd Real-Time-Face-Emotion-Detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
python main.py
```

---

## 🖥️ How It Works

1. Capture frames from your webcam using OpenCV.
2. Detect faces using a Haar Cascade or DNN-based face detector.
3. Preprocess each detected face (resize, grayscale, normalize).
4. Feed the preprocessed face into the CNN model.
5. Display the detected emotion label on the live video feed in real time.

---

## 📈 Model Evaluation

* **Metrics:** Accuracy, Loss, Confusion Matrix
* **Visualization:** Training & validation plots generated using Matplotlib

Example:

```python
# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()
```

---

## 📸 Sample Output

| Emotion     | Live Detection Example           |
| ----------- | -------------------------------- |
| Happy 😊    | ![happy](assets/happy.png)       |
| Sad 😔      | ![sad](assets/sad.png)           |
| Surprise 😲 | ![surprise](assets/surprise.png) |

*(Add your screenshots or GIF demos here)*

---

## 🔧 Future Improvements

* Upgrade model with **Transfer Learning (ResNet / MobileNet)** for higher accuracy.
* Add support for **multiple faces** in one frame.
* Integrate **Flask or Streamlit** for web-based deployment.

---

## 🧾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 💡 Acknowledgments

* FER2013 Dataset (Kaggle)
* TensorFlow & OpenCV Documentation
* Inspiration from open-source emotion recognition projects

---



