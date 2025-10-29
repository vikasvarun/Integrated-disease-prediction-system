import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
file_path = r"C:\Users\sudeep\Desktop\m2\parkinsons.data"
df = pd.read_csv(file_path)
df = df.drop(['name'], axis=1)
X = df.drop(['status'], axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Parkinson’s Model Accuracy:", accuracy)
joblib.dump(model, "parkinsons_model.pkl")
