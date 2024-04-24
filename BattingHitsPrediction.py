import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Read the data
file_path = r"C:\Users\zack3\OneDrive\Desktop\BaseballStats\lahman_1871-2023_csv\Batting.csv"
batting_data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Remove rows with missing values
batting_data.dropna(inplace=True)

# Step 3: Select features
X = batting_data[['G', 'AB', 'R', '2B', '3B', 'HR', 'RBI', 'BB', 'SO']]
y = batting_data['H']

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict the number of hits for the entire dataset
predicted_hits = model.predict(X_scaled)

# Step 8: Evaluate the model
mse = mean_squared_error(y, predicted_hits)
rmse = mse ** 0.5
r2 = r2_score(y, predicted_hits)

print("\nEvaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Step 9: Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(batting_data.index, batting_data['H'], label='Actual Hits', color='blue')
plt.plot(batting_data.index, predicted_hits, label='Predicted Hits', color='red')
plt.xlabel('Data Point Index')
plt.ylabel('Hits')
plt.title('Actual vs. Predicted Hits')
plt.legend()
plt.grid(True)
plt.show()

# Explanation of the model used
print("\nExplanation:")
print("I used a Random Forest Regressor model to predict the number of hits. Random Forest is an ensemble learning method that combines multiple decision trees, which can capture complex relationships in the data.")
