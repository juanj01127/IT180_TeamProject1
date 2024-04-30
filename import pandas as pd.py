import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Read the spreadsheet
file_path = r'C:\Users\raymo\OneDrive\Batting.xlsx'

# Load the Excel file into a pandas DataFrame
batting_data = pd.read_excel(file_path)

# Step 2: Prepare the data
# Filter data for the years 1973 to 2023 excluding specific years
years_to_exclude = [2020, 1995, 1994, 1981, 1990]
batting_data = batting_data.loc[batting_data['yearID'].between(1973, 2023) & ~batting_data['yearID'].isin(years_to_exclude)]

# Step 3: Find the player with the most hits each year
max_hits_by_year = batting_data.groupby('yearID').apply(lambda x: x.loc[x['H'].idxmax()]).reset_index(drop=True)

# Step 4: Make a prediction model for the player with the most hits each year
# For simplicity, let's use linear regression for prediction
X = max_hits_by_year[['yearID']].values.reshape(-1, 1)  # Features: Year
y = max_hits_by_year['H'].values  # Target variable: Hits

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Step 5: Predicting the amount of hits for the player with the most hits each year
# Predict hits for the years 1973 to 2023
predicted_hits = model.predict(X)

# Step 6: Generate a graph of the actual hits and predicted hits
plt.figure(figsize=(10, 6))
plt.plot(max_hits_by_year['yearID'], y, label='Actual Hits', marker='o')
plt.plot(max_hits_by_year['yearID'], predicted_hits, label='Predicted Hits', linestyle='--', marker='x')
plt.title('Actual vs Predicted Hits for Player with Most Hits Each Year (1973-2023, Excluding 2020, 1995, 1994, 1981, 1990)')
plt.xlabel('Year')
plt.ylabel('Hits')
plt.xticks(max_hits_by_year['yearID'], rotation=45)  # Set x-axis ticks to be the years
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
