# Predictive Analysis for Manufacturing Operations

## Project Overview

This project implements a machine learning-powered RESTful API for predicting machine downtime and potential production defects in manufacturing operations. By leveraging advanced predictive analytics, the solution aims to enhance operational efficiency and proactive maintenance strategies.

## Key Features

- **Data Upload**: Seamless CSV file upload for manufacturing datasets
- **Model Training**: Automated machine learning model training
- **Predictive Insights**: Real-time prediction of machine downtime with confidence scores
- **Scalable Architecture**: Built using FastAPI and scikit-learn

## Technical Stack

- **Backend**: FastAPI
- **Machine Learning**: scikit-learn
- **Data Handling**: pandas
- **Model Serialization**: joblib
- **Language**: Python 3.10+

## Project Structure

```
predictive-manufacturing-api/
│
├── main.py                 # Main FastAPI application
├── utils/
│   ├── data_handler.py     # Data processing utilities
│   └── model_handler.py    # Machine learning model operations
├── static/                 # Uploaded data storage
├── model_store/            # Trained model storage
└── requirements.txt        # Project dependencies
```

## Prerequisites

- Python 3.10+
- pip (Python Package Manager)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/manufacturing-predictive-analysis.git
   cd manufacturing-predictive-analysis
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Dataset
- **Endpoint**: `POST /upload`
- **Purpose**: Upload manufacturing dataset for model training
- **Accepts**: CSV file

### 2. Train Model
- **Endpoint**: `POST /train`
- **Purpose**: Train machine learning model on uploaded dataset
- **Returns**: Model performance metrics (accuracy, F1-score)

### 3. Predict Downtime
- **Endpoint**: `POST /predict`
- **Purpose**: Predict machine downtime based on input features
- **Returns**: 
  ```json
  {
    "Downtime": "Yes/No",
    "Confidence": 0.85
  }
  ```

## Example Request (Prediction)

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
         "Product_ID": "MACH001",
         "UDI": 1.0,
         "Type": "L",
         "Air_temperature": 300.2,
         "Process_temperature": 310.5,
         "Rotational_speed": 1500.0,
         "Torque": 45.5,
         "Tool_wear": 20.0,
         "TWF": 0.0,
         "HDF": 0.0,
         "PWF": 0.0,
         "OSF": 0.0,
         "RNF": 0.0
     }'
```

## Model Details

- **Algorithm**: Logistic Regression
- **Training**: Supervised learning
- **Performance Metrics**: Accuracy, F1-Score

## Error Handling

The API provides comprehensive error responses for scenarios like:
- Missing training data
- Invalid input formats
- Model training failures

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Email - yashkb1004@gmail.com

Project Link: https://github.com/yashBorkar404/manufacturing-predictive-analysis
