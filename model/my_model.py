import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

file_path = "C:\\Users\\tyagi\\Downloads\\pima-indians-diabetes.csv.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(file_path, skiprows=1, names=column_names)
print(data.head())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaler_path = 'C:\\Users\\tyagi\\Desktop\\my_scaler.pkl'
joblib.dump(scaler, scaler_path)

my_model = RandomForestClassifier(random_state=42)

my_model.fit(X, y)

pregnancies = float(input("Enter the number of pregnancies: "))
glucose = float(input("Enter the glucose level: "))
blood_pressure = float(input("Enter the blood pressure: "))
skin_thickness = float(input("Enter the skin thickness: "))
insulin = float(input("Enter the insulin level: "))
bmi = float(input("Enter the BMI: "))
pedigree_function = float(input("Enter the diabetes pedigree function: "))
age = float(input("Enter the age: "))

user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]],
                         columns=column_names[:-1])

user_data = scaler.transform(user_data)

prediction = my_model.predict(user_data)

if prediction[0] == 0:
    print("Based on the input, it is predicted that the person does not have diabetes.")
else:
    print("Based on the input, it is predicted that the person has diabetes.")
directory = r'C:\Users\tyagi\Desktop'
if not os.path.exists(directory):
    os.makedirs(directory)

model_path = os.path.join(directory, 'my_model.pkl')
joblib.dump(my_model, model_path)