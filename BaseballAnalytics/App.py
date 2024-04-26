from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the People table into a pandas dataframe
people_df = pd.read_excel("C:\\Users\\JCuellar\\Downloads\\People.xlsx")
teams_df = pd.read_excel("C:\\Users\\JCuellar\\Downloads\\Teams.xlsx")

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


if __name__ == '__main__':
    app.run(debug=True)
