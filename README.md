# 🥛 Milk Quality Classifier

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32.0%2B-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**AI-Powered Milk Quality Analysis System**

*Leveraging advanced machine learning to predict milk quality grades based on physicochemical and sensory parameters*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Performance Metrics](#-performance-metrics)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The **Milk Quality Classifier** is a sophisticated machine learning application that predicts milk quality grades (Low, Medium, High) based on multiple input parameters. Built with XGBoost and deployed using Streamlit, this tool provides real-time quality assessment for milk samples, making it valuable for dairy farms, quality control laboratories, and food safety inspectors.

### Why This Project?

- **Food Safety**: Ensures milk quality meets safety standards
- **Efficiency**: Instant predictions compared to traditional lab testing
- **Accuracy**: Machine learning model trained on real-world data
- **User-Friendly**: Intuitive web interface requiring no technical expertise

---

## ✨ Features

### 🎨 **Professional UI/UX Design**
- Modern dark glassmorphism theme with gradient backgrounds
- Smooth animations and transitions
- Responsive design for all screen sizes
- Interactive hover effects and visual feedback

### 🤖 **Advanced Machine Learning**
- XGBoost gradient boosting classifier
- Multi-factor analysis (7 input parameters)
- High accuracy predictions with confidence scores
- Tuned hyperparameters for optimal performance

### 📊 **Comprehensive Analysis**
- Real-time quality grading (Low/Medium/High)
- Confidence distribution visualization
- Detailed parameter breakdown
- Color-coded alerts and status indicators

### ⚡ **Performance**
- Fast prediction response time
- Cached model loading for efficiency
- Optimized for production deployment

---

## 🎥 Demo

### Application Interface

The application features a stunning dark glassmorphism design with:
- **Banner**: Professional AI-powered analysis branding
- **Sidebar**: Parameter configuration with intuitive sliders and dropdowns
- **Main Panel**: Real-time predictions with detailed confidence metrics
- **Results**: Visual quality grading with color-coded indicators

### Sample Prediction

**Input Parameters:**
- pH: 6.6
- Temperature: 45°C
- Taste: Good
- Odor: Good
- Fat: Optimal
- Turbidity: Low
- Colour: 254

**Output:**
- Quality Grade: **High**
- Confidence: **95.2%**
- Status: **Excellent ✨**

---

## 🛠 Technology Stack

### Frontend
- **Streamlit** (v1.32.0+) - Web application framework
- **HTML/CSS** - Custom styling and animations
- **Google Fonts** - Inter & Poppins typography

### Backend & ML
- **Python** (v3.8+) - Core programming language
- **XGBoost** (v2.0.3) - Gradient boosting classifier
- **scikit-learn** (v1.4.0+) - Model evaluation and preprocessing
- **pandas** (v2.2.2+) - Data manipulation
- **NumPy** (v1.26.4+) - Numerical computations

### Model Serialization
- **joblib** - Model persistence

---

## 💻 Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd XGBClassifier-main
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python --version
   streamlit --version
   ```

---

## 🚀 Usage

### Training the Model

Before running the application, train the XGBoost model:

```bash
python model.py
```

**What this does:**
- Loads the milk quality dataset (`milk_quality_data.csv`)
- Preprocesses and encodes the data
- Trains multiple classifiers (Logistic Regression, Decision Tree, Gradient Boosting, XGBoost)
- Saves the best-performing model as `model.pkl`
- Displays accuracy metrics and confusion matrix

**Sample Output:**
```
Logistic Regression Accuracy: 0.9623
Decision Tree Accuracy: 0.9547
Gradient Boosting Accuracy: 0.9811
XGBoost Accuracy: 0.9849
XGBoost (Tuned) Accuracy: 0.9906

Model saved to model.pkl
```

### Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Making Predictions

1. **Configure Parameters** in the sidebar:
   - Adjust physical properties (pH, Temperature, Colour)
   - Select chemical properties (Fat content)
   - Choose sensory properties (Taste, Odor)
   - Set turbidity level

2. **Click "🚀 Analyze Quality"** button

3. **View Results**:
   - Quality grade prediction
   - Confidence percentage
   - Status indicator
   - Detailed parameter breakdown
   - Probability distribution

---

## 🧠 Model Architecture

### Algorithm: XGBoost (Extreme Gradient Boosting)

**Why XGBoost?**
- Superior performance on tabular data
- Handles non-linear relationships effectively
- Built-in regularization prevents overfitting
- Fast training and prediction

### Model Configuration

```python
XGBClassifier(
    n_estimators=100,      # Number of boosting rounds
    max_depth=5,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    colsample_bytree=0.5,  # Feature sampling ratio
    random_state=42        # Reproducibility
)
```

### Input Features (7 Parameters)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| **pH** | Float | 3.0 - 9.5 | Acidity/alkalinity level |
| **Temperature** | Integer | 20 - 90°C | Storage/processing temperature |
| **Taste** | Binary | 0/1 | Bad (0) or Good (1) |
| **Odor** | Binary | 0/1 | Bad (0) or Good (1) |
| **Fat** | Binary | 0/1 | Not Optimal (0) or Optimal (1) |
| **Turbidity** | Binary | 0/1 | Low (0) or High (1) |
| **Colour** | Integer | 240 - 260 | Color measurement value |

### Output Classes

- **0 - Low Quality**: Below acceptable standards
- **1 - Medium Quality**: Acceptable but improvable
- **2 - High Quality**: Excellent quality standards

---

## 📊 Dataset Information

### Source
The model is trained on `milk_quality_data.csv`, a comprehensive dataset containing real-world milk quality measurements.

### Dataset Characteristics
- **Total Samples**: 1,059 records
- **Features**: 7 input parameters + 1 target variable
- **Target Distribution**: Balanced across Low, Medium, and High grades
- **Missing Values**: None (complete dataset)

### Data Preparation
1. **Encoding**: Target variable mapped (low→0, medium→1, high→2)
2. **Splitting**: 70% training, 30% testing
3. **No Scaling**: XGBoost naturally handles different scales

---

## 📁 Project Structure

```
XGBClassifier-main/
│
├── app.py                      # Streamlit web application
├── model.py                    # Model training script
├── model.pkl                   # Trained XGBoost model (generated)
├── milk_quality_data.csv       # Training dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

### File Descriptions

- **`app.py`**: Main Streamlit application with professional UI/UX design
- **`model.py`**: Training pipeline for XGBoost and comparison models
- **`model.pkl`**: Serialized trained model (created after running `model.py`)
- **`milk_quality_data.csv`**: Dataset with physicochemical and sensory parameters
- **`requirements.txt`**: All required Python packages with versions

---

## 🔌 API Reference

### Model Input Format

```python
import pandas as pd

input_data = pd.DataFrame({
    'ph': [6.6],
    'temperature': [45],
    'taste': [1],          # 0 = Bad, 1 = Good
    'odor': [1],           # 0 = Bad, 1 = Good
    'fat': [1],            # 0 = Not Optimal, 1 = Optimal
    'turbidity': [0],      # 0 = Low, 1 = High
    'colour': [254]
})
```

### Making Predictions

```python
from joblib import load

# Load trained model
model = load('model.pkl')

# Predict quality grade
prediction = model.predict(input_data)[0]  # Returns: 0, 1, or 2

# Get confidence scores
probabilities = model.predict_proba(input_data)[0]
# Returns: [prob_low, prob_medium, prob_high]

# Example output
print(f"Predicted Grade: {prediction}")
print(f"Confidence: {probabilities[prediction] * 100:.2f}%")
```

### Output Interpretation

```python
grade_mapping = {
    0: "Low Quality",
    1: "Medium Quality",
    2: "High Quality"
}
```

---

## 📈 Performance Metrics

### Model Comparison Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 96.23% | Baseline linear model |
| Decision Tree | 95.47% | Simple tree-based |
| Gradient Boosting | 98.11% | Ensemble method |
| **XGBoost (Default)** | **98.49%** | Advanced boosting |
| **XGBoost (Tuned)** | **99.06%** | ✅ **Best Performance** |

### Classification Metrics

For the tuned XGBoost model:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Low | 0.99 | 0.98 | 0.99 |
| Medium | 0.99 | 0.99 | 0.99 |
| High | 0.99 | 1.00 | 0.99 |

**Overall Accuracy**: 99.06%

---

## 🚀 Future Enhancements

### Potential Improvements

- [ ] **Batch Processing**: Upload CSV files for multiple predictions
- [ ] **Historical Analysis**: Track and visualize quality trends over time
- [ ] **Export Reports**: Generate PDF/Excel quality assessment reports
- [ ] **Mobile App**: Native iOS/Android application
- [ ] **API Endpoint**: RESTful API for integration with other systems
- [ ] **Advanced Visualizations**: 3D plots, correlation heatmaps
- [ ] **User Authentication**: Multi-user support with role-based access
- [ ] **Database Integration**: Store predictions and user data
- [ ] **Model Retraining**: Automated retraining with new data
- [ ] **Explainability**: SHAP values for feature importance

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add comments and docstrings
- Update documentation for new features
- Test thoroughly before submitting
- Keep commits atomic and well-described

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Ayyapparaja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Contact

**Ayyapparaja**

- GitHub: [@ayyapparaja227-arch](https://github.com/ayyapparaja227-arch)
- Project Link: [Milk Quality Classifier](#)

---

## 🙏 Acknowledgments

- **XGBoost Team** - For the powerful gradient boosting library
- **Streamlit** - For the intuitive web app framework
- **scikit-learn** - For comprehensive ML tools
- **Open Source Community** - For continuous inspiration and support

---

<div align="center">

**Made with ❤️ by Ayyapparaja**

*Empowering food safety through artificial intelligence*

⭐ **Star this repository if you found it helpful!** ⭐

</div>

