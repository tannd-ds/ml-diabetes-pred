import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = [] 
BOOLEAN_FEATURES = []
NUMERICAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI', 
    'DiabetesPedigreeFunction', 'Age'
]
ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES 

def create_dummy_data(num_samples=200):
    """ For dev purposes only. """
    data = {}
    for feature in NUMERICAL_FEATURES:
        data[feature] = np.random.rand(num_samples) * 100 
        if feature == 'Age': data[feature] = np.random.randint(20, 80, num_samples)
        if feature == 'BMI': data[feature] = np.random.uniform(18, 40, num_samples)

    for feature in BOOLEAN_FEATURES:
        data[feature] = np.random.choice([True, False, None], size=num_samples, p=[0.4, 0.4, 0.2]) # Allow Nones

    target = (data['Glucose'] > 70) & (data['BMI'] > 25) | (data['Age'] > 50)
    df = pd.DataFrame(data)
    df['Outcome'] = target.astype(int)
    return df

def main():
    print("Generating dummy data...")
    df = create_dummy_data(num_samples=500)

    for bf in BOOLEAN_FEATURES:
        df[bf] = df[bf].fillna(False).astype(int)
    
    for nf in NUMERICAL_FEATURES:
        if df[nf].isnull().any():
            df[nf] = df[nf].fillna(df[nf].median())

    X = df[NUMERICAL_FEATURES + BOOLEAN_FEATURES]
    y = df['Outcome']

    print("Defining preprocessing and model pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bool', 'passthrough', BOOLEAN_FEATURES)
        ],
        remainder='drop' # Drop other columns
    )

    model = LogisticRegression(solver='liblinear', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])


if __name__ == "__main__":
    main() 