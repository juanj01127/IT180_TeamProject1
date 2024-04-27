from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the People table into a pandas dataframe
people_df = pd.read_excel(r"BaseballAnalytics\Database\People.xlsx")
teams_df = pd.read_excel(r"BaseballAnalytics\Database\Teams.xlsx")
pitching_df = pd.read_excel(r"BaseballAnalytics\Database\Pitching.xlsx")
fielding_df = pd.read_excel(r"BaseballAnalytics\Database\Fielding.xlsx")
batting_df = pd.read_excel(r"BaseballAnalytics\Database\Batting.xlsx")


# Merge datasets on playerID
batting_merged = pd.merge(batting_df, people_df, on="playerID")
pitching_merged = pd.merge(pitching_df, people_df, on="playerID")

# Calculate age for each season played
batting_merged['age'] = batting_merged['yearID'] - batting_merged['birthYear']
pitching_merged['age'] = pitching_merged['yearID'] - pitching_merged['birthYear']

# Function to calculate performance trends
def calculate_performance_trends():
    # Aggregate statistics by age for batting and pitching
    batting_stats_by_age = batting_merged.groupby('age').agg({'H': 'sum', 'HR': 'sum', 'AB': 'sum'})
    pitching_stats_by_age = pitching_merged.groupby('age').agg({'SO': 'sum', 'ERA': 'mean'})

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Batting Statistics
    axes[0].plot(batting_stats_by_age.index, batting_stats_by_age['H'], label='Hits')
    axes[0].plot(batting_stats_by_age.index, batting_stats_by_age['HR'], label='Home Runs')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Batting Performance by Age')
    axes[0].legend()

    # Pitching Statistics
    axes[1].plot(pitching_stats_by_age.index, pitching_stats_by_age['SO'], label='Strikeouts')
    axes[1].plot(pitching_stats_by_age.index, pitching_stats_by_age['ERA'], label='ERA')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count/Value')
    axes[1].set_title('Pitching Performance by Age')
    axes[1].legend()

    # Convert plot to HTML
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_data

# Function to display all player information for a given season
def display_players_for_season(year):
    # Filter the dataframe for the given season
    players_for_season = people_df[people_df['birthYear'] == year]
    
    # Check if any players are found for the given season
    if players_for_season.empty:
        return None
    else:
        # Construct a list of strings containing player information
        players_info = []
        for index, player in players_for_season.iterrows():
            player_info = f"{player['nameFirst']} {player['nameLast']} - Birth Date: {player['birthYear']}-{player['birthMonth']}-{player['birthDay']}, Birth Place: {player['birthCity']}, {player['birthCountry']}, {player['birthState']}"
            players_info.append(player_info)
        return players_info
    
def display_teams_for_season(year):
    # Filter the dataframe for the given season
    teams_for_season = teams_df[teams_df['yearID'] == year]
    
    # Check if any teams are found for the given season
    if teams_for_season.empty:
        return None
    else:
        # Construct a list of strings containing team names, wins, and losses
        teams_info = []
        for index, team in teams_for_season.iterrows():
            team_info = f"{team['name']} - Wins: {team['W']}, Losses: {team['L']}"
            teams_info.append(team_info)
        return teams_info
    
# Route for displaying performance trends
@app.route('/performance_trends', methods=['GET', 'POST'])
def performance_trends():
    if request.method == 'POST':
        plot_data = calculate_performance_trends()
        return render_template('performance_trends.html', plot_data=plot_data)
    return render_template('performance_trends.html', plot_data=None)


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/players', methods=['GET', 'POST'])
def display_players():
    if request.method == 'POST':
        year = int(request.form['year'])  # Get year from form submission
        players_info = display_players_for_season(year)
        if players_info is None:
            return render_template('display_players.html', year=year, error=True)
        else:
            return render_template('display_players.html', year=year, players_info=players_info)
    return render_template('display_players.html', error=False)

# Route for displaying teams for a given season
@app.route('/teams', methods=['GET', 'POST'])
def display_teams():
    if request.method == 'POST':
        year = int(request.form['year'])  # Get year from form submission
        teams_info = display_teams_for_season(year)
        if teams_info is None:
            return render_template('display_teams.html', year=year, error=True)
        else:
            return render_template('display_teams.html', year=year, teams_info=teams_info)
    return render_template('display_teams.html', error=False)

# Calculate batting average (AVG) and add it to the Batting DataFrame
batting_df['AVG'] = batting_df['H'] / batting_df['AB']

# Merge data tables
merged_df = pd.merge(batting_df, people_df, on='playerID', how='left')

# Calculate age based on 'yearID' and 'birthYear'
merged_df['age'] = merged_df['yearID'] - merged_df['birthYear']

# Handle missing values for numeric columns only
numeric_cols = merged_df.select_dtypes(include=['number']).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())


# Feature Engineering
features = ['yearID', 'age', 'AB', 'H']
X = merged_df[features]
y = merged_df['AVG']  # Target variable (e.g., batting average)

# Model Selection and Training
model = LinearRegression()
model.fit(X, y)

# Route for predicting batting average
@app.route('/predict_batting_avg', methods=['GET', 'POST'])
def predict_batting_avg():
    if request.method == 'POST':
        player_id = request.form['player_id']  # Get player ID from form submission

        # Filter data for the specified player ID
        player_data = merged_df[merged_df['playerID'] == player_id]

        # Predict future performance for the specific player
        future_performance = model.predict(player_data[features])

        # Round the predicted batting average to three decimal places
        rounded_avg = round(future_performance[0], 3)

        return render_template('predict_batting_avg.html', player_id=player_id, predicted_avg=rounded_avg)
    
    return render_template('predict_batting_avg.html', player_id=None, predicted_avg=None)


if __name__ == '__main__':
    app.run(debug=True)
