import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Loading
people_df = pd.read_csv("people.csv")
batting_df = pd.read_csv("batting.csv")
pitching_df = pd.read_csv("pitching.csv")
fielding_df = pd.read_csv("fielding.csv")
teams_df = pd.read_csv("teams.csv")

# Step 2: Data Cleaning and Preparation
# We'll assume the data is clean for now since it's directly from CSV files

# Step 3: Feature Engineering
# We'll focus on predicting batting average for the next season
# So, we need to calculate batting average and consider other relevant features
batting_df['BA'] = batting_df['H'] / batting_df['AB']  # Batting Average
batting_features = ['playerID', 'yearID', 'BA']
batting_features_df = batting_df[batting_features]

# Step 4: Merge Data
# Merge batting features with people data to get player information
player_batting_df = pd.merge(batting_features_df, people_df, on='playerID', how='left')

# Step 5: Prepare Data for Modeling
# We'll use previous seasons' stats, age, and team performance metrics as features
# For simplicity, we'll focus on a single team performance metric: team batting average
team_batting_avg = teams_df.groupby('yearID')['H'].sum() / teams_df.groupby('yearID')['AB'].sum()
team_batting_avg = team_batting_avg.reset_index()
team_batting_avg.columns = ['yearID', 'Team_BA']
player_batting_df = pd.merge(player_batting_df, team_batting_avg, on='yearID', how='left')

# Step 6: Model Building
# Split data into training and testing sets
X = player_batting_df[['BA', 'Team_BA', 'age']]
y = player_batting_df['BA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Evaluate the model's performance on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Prediction
# Predict batting average for the next season
# For demonstration purposes, we'll use the same features for prediction
# In practice, we'd use the most recent season's data for prediction
future_batting_df = pd.DataFrame({
    'BA': [0.300],  # Example previous season's batting average
    'Team_BA': [0.280],  # Example team batting average for the previous season
    'age': [30]  # Example player's age
})
future_prediction = model.predict(future_batting_df)
print("Predicted Batting Average for Next Season:", future_prediction[0])

# Step 9: Model Validation
# Validate the model's predictions against actual future performance data
# This step requires access to future data, which is not provided in the current dataset
# In practice, we'd compare the model's predictions with actual performance data for validation

# Step 10: Model Deployment
# Once the model is validated and deemed satisfactory, it can be deployed for making predictions in real-world scenarios

# Optional: Further Analysis and Improvement
# Depending on the specific requirements and performance of the model, further analysis and improvement steps may be taken
