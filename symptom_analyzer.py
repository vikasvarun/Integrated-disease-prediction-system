from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pandas as pd

# Corrected path reading
df = pd.read_csv(r"C:\Users\sudeep\Desktop\m2\Disease and symptoms dataset.csv")

# Assuming the first column is disease and the rest are symptoms
symptom_columns = df.columns[1:]

# Create a list of symptoms for each row
df['Symptom'] = df[symptom_columns].apply(lambda row: [symptom_columns[i] for i in range(len(row)) if row.iloc[i] == 1], axis=1)

# Fit the label binarizer
mlb = MultiLabelBinarizer()
mlb.fit(df['Symptom'])

# Save the binarizer
joblib.dump(mlb, r"C:\Users\sudeep\Desktop\m2\label_binarizer.pkl")

print("✅ label_binarizer.pkl created successfully!")
