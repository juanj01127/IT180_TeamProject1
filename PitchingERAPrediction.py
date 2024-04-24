import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Read the data
file_path = r"C:\Users\zack3\OneDrive\Desktop\BaseballStats\lahman_1871-2023_csv\Pitching.csv"
pitching_data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Remove rows with missing values
pitching_data.dropna(inplace=True)

# Step 3: Select features
features = ['W', 'L', 'G', 'H', 'ER', 'HR', 'BB', 'SO', 'BAOpp']

# Step 4: Train a predictive model
X = pitching_data[features]
y = pitching_data['ERA']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict the ERA for the entire dataset
predicted_era = model.predict(X_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y, predicted_era)
rmse = mse ** 0.5
r2 = r2_score(y, predicted_era)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Additional statistics from the model
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
print("\nFeature Importance:")
print(feature_importance)

# Step 6: Visualize the results
plt.figure(figsize=(12, 6))

# Plot actual ERA
plt.plot(pitching_data.index, pitching_data['ERA'], label='Actual ERA', color='blue')

# Plot predicted ERA
plt.plot(pitching_data.index, predicted_era, label='Predicted ERA', color='red')

plt.xlabel('Data Point Index')
plt.ylabel('ERA')
plt.title('Actual vs. Predicted ERA')
plt.legend()
plt.grid(True)
plt.show()

# Explanation of the model used
print("\nExplanation:")
print("I used a Random Forest Regressor model for predicting ERA. Random Forest is an ensemble learning method that combines multiple decision trees, which can capture complex relationships in the data.")
