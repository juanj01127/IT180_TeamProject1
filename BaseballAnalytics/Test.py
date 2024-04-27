import pandas as pd
import matplotlib.pyplot as plt

# Load data from provided datasets
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

plt.tight_layout()
plt.show()


#This code will calculate the age of each player for each season played, aggregate statistics by age for batting (hits and home runs) and pitching (strikeouts and ERA), and plot the trends of these statistics against age to identify peak performance periods. You can further customize the analysis by adding more statistical categories or refining the visualization as needed.