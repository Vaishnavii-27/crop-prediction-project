import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('data/crop_data.csv')

print(df.head())  # check data

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model/model.pkl', 'wb'))

print("✅ Model trained and saved!")