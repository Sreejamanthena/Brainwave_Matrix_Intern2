from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from utils.preprocessing import load_and_preprocess_data

def main():
    print("Loading and preprocessing data...")

    # Load and preprocess
    X, y = load_and_preprocess_data("data/creditcard.csv")
    print("Data preprocessed. Total samples after SMOTE:", len(X))

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained. Evaluating...")

    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Save the model
    joblib.dump(model, "models/fraud_model.pkl")
    print("Model saved to models/fraud_model.pkl")

if __name__ == "__main__":
    main()
