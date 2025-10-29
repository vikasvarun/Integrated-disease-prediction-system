import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv(r"C:\Users\sudeep\Desktop\m2\Diabetes Predictions.csv") 
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(model, "diabetes_model.pkl")
