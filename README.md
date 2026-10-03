# 🧠 Task 6: Creating a Streamlit User Interface for Deep Learning Prediction

## 📌 Project Overview

This project implements a user-friendly **Streamlit web application** for interacting with a trained deep learning image classification model.

The application uses a previously developed **CIFAR-10 image classification model** saved as:

```text
cifar10_model.keras
```

Users can upload an image through the Streamlit interface, and the trained deep learning model predicts which CIFAR-10 category the image belongs to.

---

## 🎯 Objective

The objective of Task 6 is to:

- Build a user-friendly web interface using Streamlit.
- Allow users to upload images.
- Preprocess uploaded images according to the model requirements.
- Connect the Streamlit interface with the trained deep learning model.
- Display prediction results.
- Display prediction confidence.
- Visualize probabilities for all CIFAR-10 classes.
- Improve usability using Streamlit layout components.

---

## 📊 Dataset

The application is based on the **CIFAR-10 dataset**.

CIFAR-10 contains **10 image classes**:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The original CIFAR-10 images have a resolution of **32 × 32 pixels** with three RGB color channels.

---

## 🤖 Deep Learning Model

The trained model is stored in:

```text
cifar10_model.keras
```

### Model Framework

- TensorFlow
- Keras

### Input

```text
32 × 32 × 3
```

### Output

The model predicts probabilities for the 10 CIFAR-10 classes.

The class with the highest probability is selected as the final prediction.

---

## 🖥️ Streamlit Application

The application provides an interactive interface where users can:

- Upload JPG, JPEG, or PNG images.
- View the uploaded image.
- View image information.
- Preprocess the image automatically.
- Run the trained deep learning model.
- View the predicted class.
- View prediction confidence.
- View probabilities for all 10 classes.
- View a probability bar chart.
- View the top 3 predictions.

---

## 🔄 Application Workflow

```text
User Uploads Image
        ↓
Image Validation
        ↓
Image Resizing
        ↓
Pixel Normalization
        ↓
Deep Learning Model
        ↓
Prediction Probabilities
        ↓
Predicted CIFAR-10 Class
        ↓
Confidence & Visualization
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| TensorFlow | Deep learning model |
| Keras | Model loading and prediction |
| Streamlit | Web user interface |
| NumPy | Numerical processing |
| Pandas | Prediction data handling |
| Pillow | Image processing |
| Matplotlib | Prediction visualization |

---

## 📁 Project Structure

```text
Task_6_Streamlit/
│
├── app.py
├── cifar10_model.keras
├── requirements.txt
└── README.md
```

### File Description

**`app.py`**

Contains the complete Streamlit application and prediction logic.

**`cifar10_model.keras`**

Contains the trained CIFAR-10 deep learning model.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

**`README.md`**

Contains project documentation and instructions.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone <https://github.com/228263satyam/Task-_6_Streamlit.git>
```

Navigate to the project directory:

```bash
cd Task_6_Streamlit
```

---

### Step 2: Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
streamlit
tensorflow
numpy
pandas
pillow
matplotlib
```

---

## ▶️ Running the Application

Run the following command:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

The default local address is:

```text
http://localhost:8501
```

---

## 🖼️ Using the Application

### Step 1

Open the Streamlit application.

### Step 2

Click:

```text
Browse files
```

and upload an image.

### Step 3

The application displays the uploaded image and image information.

### Step 4

Click:

```text
🚀 Predict Image
```

### Step 5

The deep learning model processes the image and displays:

- Predicted class
- Confidence percentage
- Probability distribution
- Top 3 predictions
- Probability visualization

---

## 📈 Output Example

The application produces an output similar to:

```text
Predicted Class: DOG

Confidence: 94.52%
```

It also displays the probability distribution across all 10 CIFAR-10 classes.

---

## 🎨 User Interface Features

The Streamlit interface includes:

- Clean dashboard-style layout
- Sidebar with model information
- Image upload widget
- Image preview
- Prediction button
- Prediction result cards
- Confidence display
- Probability table
- Probability chart
- Top-3 predictions
- User-friendly messages
- Responsive column-based layout

---

## 🧪 Model Prediction Process

The uploaded image is converted to RGB format and resized to:

```text
32 × 32 pixels
```

Pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

The processed image is then passed to the trained Keras model.

The predicted class is obtained using the highest probability:

```python
predicted_index = np.argmax(predictions[0])
```

---

## 📋 Evaluation Criteria

The project addresses the required evaluation criteria:

### 1. Functionality

The application successfully:

- Accepts user images.
- Loads the trained model.
- Performs image preprocessing.
- Generates predictions.
- Displays prediction results.

### 2. User Experience

The interface provides:

- Simple navigation
- Clear instructions
- Image preview
- Prediction feedback
- Confidence information
- Visual probability results

### 3. Interface Design Quality

The application uses:

- Streamlit columns
- Sidebar navigation
- Metrics
- Tables
- Charts
- Progress indicators
- Informational messages

---

## 📦 Deliverables

The following deliverables are included:

- ✅ Streamlit application
- ✅ Trained deep learning model
- ✅ Source code
- ✅ Requirements file
- ✅ README documentation
- ✅ Application screenshots
- ✅ Prediction outputs

---

## 🚀 Future Enhancements

Possible improvements include:

- Deploying the application on Streamlit Community Cloud.
- Adding drag-and-drop image upload.
- Adding batch image prediction.
- Adding prediction history.
- Adding model performance statistics.
- Adding Grad-CAM visualization for model interpretability.
- Adding support for additional image classification models.

---

## 👨‍💻 Author

**Satyam Yadav**

M.Sc. Data Science & Big Data Analyst

---

## 📚 Academic Task

**Task 6: Creating a Streamlit User Interface**

**Objective:**  
To build a user-friendly web interface for interacting with deep learning prediction models.

**Model:** CIFAR-10 Deep Learning Image Classification Model

**Interface:** Streamlit
