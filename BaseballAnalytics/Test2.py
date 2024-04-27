import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data tables
people_df = pd.read_excel(r"BaseballAnalytics\Database\People.xlsx")
teams_df = pd.read_excel(r"BaseballAnalytics\Database\Teams.xlsx")
pitching_df = pd.read_excel(r"BaseballAnalytics\Database\Pitching.xlsx")
fielding_df = pd.read_excel(r"BaseballAnalytics\Database\Fielding.xlsx")
batting_df = pd.read_excel(r"BaseballAnalytics\Database\Batting.xlsx")

# Calculate batting average (AVG) and add it to the Batting DataFrame
batting_df['AVG'] = batting_df['H'] / batting_df['AB']

# Merge data tables
# For simplicity, let's focus on batting statistics for predictive modeling
merged_df = pd.merge(batting_df, people_df, on='playerID', how='left')

# Calculate age based on 'yearID' and 'birthYear'
merged_df['age'] = merged_df['yearID'] - merged_df['birthYear']

# Handle missing values
numeric_cols = ['yearID', 'birthYear', 'age', 'AB', 'H', 'AVG']
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())

# Feature Engineering
# Here, you can select relevant features and create additional ones if needed
# For demonstration, let's use 'yearID', 'age', 'AB' (at-bats), and 'H' (hits) as features
features = ['yearID', 'age', 'AB', 'H']
X = merged_df[features]
y = merged_df['AVG']  # Target variable (e.g., batting average)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Accept player ID as input from the user
player_id = input("Enter player ID: ")

# Filter data for the specified player ID
player_data = merged_df[merged_df['playerID'] == player_id]

# Predict Future Performance for the specific player
future_performance = model.predict(player_data[features])

# Round the batting average to three decimal places
rounded_avg = round(future_performance[0], 3)

# Display the predicted future performance
print("Predicted Batting Average for Player", player_id, "in the next season:", rounded_avg)
