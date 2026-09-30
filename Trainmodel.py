import joblib
import pandas as pd

# 1. Load your models AND your scaler
model_lr = joblib.load("lregression.pkl")
model_knn = joblib.load("knn.pkl")  
model_svm = joblib.load("svm_model.pkl")  
scaler = joblib.load("scaler.pkl")

# 2. Collect inputs from the user
pclass = int(input("Enter Pclass (1, 2, or 3): "))
age = float(input("Enter Age: "))
sibsp = int(input("Enter Siblings/Spouses Aboard (SibSp): "))
parch = int(input("Enter Parents/Children Aboard (Parch): "))
fare = float(input("Enter Fare Paid: "))
sex = input("Enter Sex (male/female): ").lower()
embarked = input("Enter Embarked Port (C/Q/S): ").upper()

# 3. Derive engineered features matching your notebook logic
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
sex_male = 1 if sex == "male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# 4. Create a DataFrame with the exact same column names and order
new_passenger = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'FamilySize': [family_size],
    'IsAlone': [is_alone],
    'Sex_male': [sex_male],
    'Embarked_Q': [embarked_q],
    'Embarked_S': [embarked_s]
})

# 5. Scale the features exactly like in your notebook
new_scaled = scaler.transform(new_passenger)

# 6. Get predictions from all three models
output_lr = model_lr.predict(new_scaled)[0]
output_knn = model_knn.predict(new_scaled)[0]
output_svm = model_svm.predict(new_scaled)[0]

# 7. Apply your working majority voting logic
total_outputs = [output_lr, output_knn, output_svm]

print("\n--- Final Consolidated Prediction ---")
if total_outputs.count(1) >= 2:
    print("Predicted: Survived")
else:
    print("Predicted: Did Not Survive")
