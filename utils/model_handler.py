from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np
from fastapi import HTTPException

def train_model(filepath: str, model_path: str):
    # Dataset is being loaded
    data = pd.read_csv(filepath)
    
    # Selecting the features for training and  labelling them
    X = data[['UDI','Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','TWF','HDF','PWF','OSF','RNF']]
    y = data['Machine failure']
    
    # Encoding categorical feature 'Type'
    label_encoder = LabelEncoder()
    X.loc[:, 'Type'] = label_encoder.fit_transform(X['Type'])
    
    # Splitting data into training and testing sets using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training Logistic Regression model(Classifying into 2 classes)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save the label encoder along with the model
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder
    }, model_path)
    
    return {'accuracy': accuracy, 'f1_score': f1}

def predict_with_model(model_path: str, X: pd.DataFrame):
    # Load the trained model and label encoder
    saved_data = joblib.load(model_path)
    model = saved_data['model']
    label_encoder = saved_data['label_encoder']
    
    # Encode the 'Type' feature
    try:
        X.loc[:, 'Type'] = label_encoder.transform(X['Type'])
    except ValueError as e:
        # Handle unseen labels by adding them to the label encoder
        unseen_labels = set(X['Type']) - set(label_encoder.classes_)
        if unseen_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, list(unseen_labels))
            X.loc[:, 'Type'] = label_encoder.transform(X['Type'])
        else:
            raise HTTPException(status_code=400, detail=f"Unseen label in 'Type' feature: {str(e)}")
    
    # Predictions
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    
    return prediction, confidence