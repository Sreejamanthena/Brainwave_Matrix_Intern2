import joblib
import pandas as pd

def predict_transaction(data_point):
    model = joblib.load("models/fraud_model.pkl")
    prediction = model.predict(data_point)  # No [ ] here!
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")
    sample = df.drop('Class', axis=1).iloc[[0]]  # double brackets keep it as DataFrame
    result = predict_transaction(sample)
    print("Prediction:", result)
