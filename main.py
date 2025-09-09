# python -m venv sklearn-env
# source sklearn-env/bin/activate  # activate
# pip install -U scikit-learn
# sudo port install py312-scikit-learn

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Load the scores dataset
scores_path = 'data/2017-2025_scores.csv'  # Adjust if filename differs
if not os.path.exists(scores_path):
    raise FileNotFoundError(f"{scores_path} not found. Please ensure the file is in the project root.")

df_scores = pd.read_csv(scores_path)

# Display basic info about the dataset
print("Scores Dataset Info:")
print(df_scores.head())
print(f"Columns: {df_scores.columns.tolist()}")
print(f"Shape: {df_scores.shape}")

# From the output, actual columns are: 'Season', 'Week', 'AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore', etc.
# Mapping: home_score -> HomeScore, away_score -> AwayScore
# Also, Week is string (e.g., 'Hall Of Fame'), so handle as categorical or extract numeric if possible
# For simplicity, we'll treat Week as categorical and add a numeric week feature if applicable

# Step 2: Preprocess scores data for game outcome prediction
# Create target: home_win (binary) - but dataset already has 'HomeWin' which is 1 if HomeTeam won
# We'll use 'HomeWin' directly as y
if 'HomeScore' not in df_scores.columns or 'AwayScore' not in df_scores.columns:
    raise ValueError("Expected 'HomeScore' and 'AwayScore' columns. Adjust column names as needed.")

# Verify home_win logic matches (optional check)
df_scores['computed_home_win'] = (df_scores['HomeScore'] > df_scores['AwayScore']).astype(int)
if not df_scores['computed_home_win'].equals(df_scores['HomeWin']):
    print("Warning: 'HomeWin' column may not match computed home win. Using computed for consistency.")
    y = df_scores['computed_home_win']
else:
    y = df_scores['HomeWin']

# Select features: adjust based on actual columns
# Use 'Season' (numeric), 'Week' (categorical for now), 'AwayTeam', 'HomeTeam'
# Add 'Day' if useful, but skip seeding/postseason for basic model (handle NaNs)
feature_cols = ['Season', 'Week', 'AwayTeam', 'HomeTeam']  # Core features
if not all(col in df_scores.columns for col in feature_cols):
    print("Warning: Some feature columns missing. Using available ones.")
    feature_cols = [col for col in feature_cols if col in df_scores.columns]

# Optional: Add numeric week extraction (e.g., map 'Preseason Week 1' to 1, etc.)
# For now, keep as is; if Week is mostly numeric, convert
try:
    df_scores['numeric_week'] = pd.to_numeric(df_scores['Week'], errors='coerce')
    if df_scores['numeric_week'].notna().sum() > df_scores.shape[0] * 0.8:  # If mostly numeric
        feature_cols.append('numeric_week')
        print("Added 'numeric_week' as feature.")
    else:
        del df_scores['numeric_week']  # Drop if not useful
        print("Week remains categorical.")
except:
    print("Week treated as categorical.")

X = df_scores[feature_cols].copy()

# Handle missing values
X = X.fillna('Unknown')  # For categorical/NaNs

# Drop rows where target is NaN if any (e.g., incomplete games)
df_scores = df_scores.dropna(subset=['HomeWin'])
X = X.loc[df_scores.index]
y = y.loc[df_scores.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical features
categorical_features = ['AwayTeam', 'HomeTeam', 'Week']  # Adjust if 'numeric_week' added
categorical_features = [col for col in categorical_features if col in feature_cols]
numerical_features = ['Season']  # Add 'numeric_week' if present
if 'numeric_week' in feature_cols:
    numerical_features.append('numeric_week')

# Preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 3: Build and train model for win prediction
win_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

win_model.fit(X_train, y_train)

# Evaluate
y_pred = win_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nHome Win Prediction Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Example prediction - adjust teams to match dataset abbreviations (e.g., 'ARI' for Cardinals, 'DAL' for Cowboys)
# From sample: 'Cardinals' -> but likely abbreviated in full data; check unique teams if needed
new_game = pd.DataFrame({
    'Season': [2025],
    'Week': ['1'],  # Or 'Regular Season Week 1'
    'AwayTeam': ['SF'],  # 49ers
    'HomeTeam': ['KC']   # Chiefs - assuming abbreviations like 'SF', 'KC'
})
if 'numeric_week' in feature_cols:
    new_game['numeric_week'] = 1
win_prob = win_model.predict_proba(new_game)[0][1]
print(f"Predicted home win probability for KC (home) vs SF (away) in 2025 Week 1: {win_prob:.2f}")

# Step 4: Extend to score prediction (regression on HomeScore, AwayScore)
# Train separate regressors for home and away scores
# Use same features/split for consistency

# For HomeScore
y_home = df_scores['HomeScore']
X_train_s, X_test_s, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42)

score_regressor_home = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

score_regressor_home.fit(X_train_s, y_train_home)
home_pred = score_regressor_home.predict(X_test_s)
mae_home = np.mean(np.abs(home_pred - y_test_home))
print(f"\nMAE for Home Score Prediction: {mae_home:.2f}")

# For AwayScore
y_away = df_scores['AwayScore']
X_train_a, X_test_a, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42)

score_regressor_away = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

score_regressor_away.fit(X_train_a, y_train_away)
away_pred = score_regressor_away.predict(X_test_a)
mae_away = np.mean(np.abs(away_pred - y_test_away))
print(f"MAE for Away Score Prediction: {mae_away:.2f}")

# Example score prediction
new_home_pred = score_regressor_home.predict(new_game)[0]
new_away_pred = score_regressor_away.predict(new_game)[0]
print(f"Predicted scores for KC vs SF: Home (KC) {new_home_pred:.1f}, Away (SF) {new_away_pred:.1f}")

# Step 5: Load and preprocess plays data for play-level predictions
# Concatenate all plays CSVs (2017-2024) - assuming they exist in project dir
plays_dfs = []
for year in range(2017, 2025):
    plays_file = f"{year}_plays.csv"
    if os.path.exists(plays_file):
        df_plays_year = pd.read_csv(plays_file)
        df_plays_year['season'] = year  # Add season if not present
        plays_dfs.append(df_plays_year)
        print(f"Loaded {plays_file}: {df_plays_year.shape}")
    else:
        print(f"{plays_file} not found, skipping.")

if plays_dfs:
    df_plays = pd.concat(plays_dfs, ignore_index=True)
    print("\nPlays Dataset Info:")
    print(df_plays.head())
    print(f"Columns: {df_plays.columns.tolist()}")
    print(f"Shape: {df_plays.shape}")

    # Assume typical play columns for NFL PBP: play_id, game_id, down, ydstogo (yards_to_go), yrdln (yardline), qtr (quarter),
    # time (game clock), posteam (offense team abbr), defteam, play_type, yards_gained, etc.
    # Adjust based on actual columns (printed above)

    # Example: Predict play_success (binary: yards_gained > 0) or yards_gained (regression)
    if 'yards_gained' in df_plays.columns:
        df_plays['play_success'] = (df_plays['yards_gained'] > 0).astype(int)
        play_y = df_plays['play_success']
        print(f"Play success defined as yards_gained > 0. Positive plays: {play_y.mean():.2%}")
    elif 'play_result' in df_plays.columns:  # Alternative column name
        df_plays['play_success'] = (df_plays['play_result'] > 0).astype(int)
        play_y = df_plays['play_success']
    else:
        print("No suitable yards column (e.g., 'yards_gained' or 'play_result'); skipping play prediction.")
        play_y = None

    if play_y is not None:
        # Select play features (adjust after checking printed columns)
        play_features = ['down', 'ydstogo', 'yrdln', 'qtr', 'time', 'posteam', 'defteam', 'season']  # Common names
        # Fallback names: ['down', 'yards_to_go', 'yardline_100', 'quarter', 'game_clock', 'posteam', 'defteam', 'season']
        play_features = [col for col in play_features if col in df_plays.columns]
        if not play_features:
            # Try alternative names
            alt_features = ['down', 'yards_to_go', 'yardline_100', 'quarter', 'game_clock', 'posteam', 'defteam', 'season']
            play_features = [col for col in alt_features if col in df_plays.columns]
            print(f"Using alternative play features: {play_features}")

        if play_features:
            X_plays = df_plays[play_features].copy()
            # Preprocess: fill NaNs, convert types
            for col in X_plays.select_dtypes(include=['object']).columns:
                X_plays[col] = X_plays[col].fillna('Unknown')
            for col in X_plays.select_dtypes(include=[np.number]).columns:
                X_plays[col] = X_plays[col].fillna(0)
            # Handle time if string (e.g., '15:00' -> seconds)
            if 'time' in X_plays.columns and X_plays['time'].dtype == 'object':
                def time_to_secs(t):
                    if pd.isna(t) or t == 'Unknown':
                        return 900  # Avg quarter
                    try:
                        m, s = map(int, str(t).split(':'))
                        return m * 60 + s
                    except:
                        return 900
                X_plays['time'] = X_plays['time'].apply(time_to_secs)

            # Drop rows with NaN target
            valid_idx = ~play_y.isna()
            X_plays = X_plays[valid_idx]
            play_y = play_y[valid_idx]

            # Split
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_plays, play_y, test_size=0.2, random_state=42)

            # Play preprocessor
            play_cat_features = [col for col in ['posteam', 'defteam'] if col in play_features]
            play_num_features = [col for col in play_features if col not in play_cat_features]

            play_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), play_num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), play_cat_features)
                ]
            )

            # Play model: classify success
            play_model = Pipeline(steps=[
                ('preprocessor', play_preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])

            play_model.fit(X_train_p, y_train_p)
            play_pred = play_model.predict(X_test_p)
            play_accuracy = accuracy_score(y_test_p, play_pred)
            print(f"\nPlay Success Prediction Accuracy: {play_accuracy:.2f}")

            # Example play prediction - adjust columns to match
            new_play_cols = {col: val for col, val in [
                ('down', 1),
                ('ydstogo', 10),  # or 'yards_to_go'
                ('yrdln', 75),    # or 'yardline_100' (opponent's 25)
                ('qtr', 1),       # or 'quarter'
                ('time', 900),    # 15:00 in secs or '15:00' if not converted
                ('posteam', 'KC'),
                ('defteam', 'SF'),
                ('season', 2025)
            ] if col in play_features}
            new_play = pd.DataFrame([new_play_cols])
            success_prob = play_model.predict_proba(new_play)[0][1]
            print(f"Predicted success probability for example play (KC offense vs SF defense): {success_prob:.2f}")
        else:
            print("No play features available; skipping.")

else:
    print("No plays data loaded. Predictions skipped.")

# Insights and trends
print("\nTeam Home Win Rates by Season (sample):")
if 'HomeTeam' in df_scores.columns and 'Season' in df_scores.columns:
    team_stats = df_scores.groupby(['HomeTeam', 'Season'])['HomeWin'].agg(['mean', 'count']).reset_index()
    team_stats = team_stats[team_stats['count'] >= 1]  # Only teams with games
    print(team_stats.head(10))
    # Overall trends
    seasonal_win_rate = df_scores.groupby('Season')['HomeWin'].mean()
    print(f"\nAverage Home Win Rate by Season:\n{seasonal_win_rate}")

# Save models (optional - uncomment if joblib installed)
# import joblib
# joblib.dump(win_model, 'win_predictor.pkl')
# joblib.dump(score_regressor_home, 'home_score_predictor.pkl')
# joblib.dump(score_regressor_away, 'away_score_predictor.pkl')
# if 'play_model' in locals():
#     joblib.dump(play_model, 'play_predictor.pkl')

# Further enhancements:
# - Feature engineering: Add team strength (e.g., rolling win % prior to game)
# - Handle ties: Dataset may have HomeWin=0 when tie; adjust if needed (e.g., 0.5 prob)
# - For Week: Create a mapping dict for numeric (Preseason 1-4 -> -3 to 0, Regular 1-18 ->1-18, etc.)
# - Hyperparameter tuning: Use GridSearchCV
# - Cross-validation: TimeSeriesSplit for seasons
# - 2024-2025 data: Since date is 2025, add/update CSVs or integrate nfl_data_py (pip install nfl_data_py)
# - Visualizations: Use matplotlib to plot win rates, score distributions