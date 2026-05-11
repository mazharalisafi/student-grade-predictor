# =============================================
# Student Grade Predictor
# By: Mazhar Ali
# Description: Predicts student grade based on
#              study hours and attendance
# =============================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------------------------
# STEP 1: Our Data
# (Study Hours, Attendance %, Final Grade)
# -----------------------------------------------
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 4, 6, 3, 5, 7, 8, 1, 9, 6],
    'attendance':  [40, 50, 55, 60, 65, 70, 75, 80, 90, 95,
                    45, 62, 71, 58, 68, 78, 85, 42, 92, 73],
    'final_grade': [35, 42, 50, 55, 62, 68, 74, 80, 88, 95,
                    40, 57, 69, 52, 64, 76, 83, 37, 91, 70]
}

# Convert data to a table (DataFrame)
df = pd.DataFrame(data)

# -----------------------------------------------
# STEP 2: Prepare Input (X) and Output (y)
# -----------------------------------------------
X = df[['study_hours', 'attendance']]  # Input features
y = df['final_grade']                  # What we want to predict

# -----------------------------------------------
# STEP 3: Split Data into Training and Testing
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# STEP 4: Train the Model
# -----------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------------------------
# STEP 5: Test the Model
# -----------------------------------------------
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Average prediction error: {error:.2f} marks")

# -----------------------------------------------
# STEP 6: Predict for a New Student
# -----------------------------------------------
print("\n--- Predict a Student's Grade ---")
study = float(input("Enter study hours per day: "))
attend = float(input("Enter attendance percentage: "))

predicted = model.predict([[study, attend]])
print(f"\nPredicted Final Grade: {predicted[0]:.1f} / 100")

# -----------------------------------------------
# STEP 7: Show a Graph
# -----------------------------------------------
plt.scatter(y_test, predictions, color='blue')
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.tight_layout()
plt.savefig("grade_results.png")
plt.show()
print("\nGraph saved as grade_results.png")