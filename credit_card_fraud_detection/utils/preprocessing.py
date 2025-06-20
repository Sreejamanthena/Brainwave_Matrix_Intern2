import pandas as pd
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Reduce size to 10% of the original dataset for faster processing
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Reduced dataset size: {len(df)} rows")

    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("SMOTE applied. New dataset size:", len(X_resampled))
    return X_resampled, y_resampled
