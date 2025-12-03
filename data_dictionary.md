# Data Dictionary — NBA 2023–24 Datasets

Columns used across the cleaned datasets. Units are per-game/per-100 as provided by the source pulls.

| Column | Type | Description |
| --- | --- | --- |
| Team | string | Team name/identifier for the game row. |
| Opponent | string | Opposing team for the game row. |
| Date | datetime | Game date (parsed from m/d/yy strings). |
| ORtg | float | Offensive Rating (points scored per 100 possessions). |
| DRtg | float | Defensive Rating (points allowed per 100 possessions). |
| NRtg | float | Net Rating (ORtg – DRtg). |
| Pace | float | Estimated possessions per 48 minutes. |
| eFG% | float | Effective Field Goal Percentage: (FG + 0.5 * 3P) / FGA. |
| TS% | float | True Shooting Percentage: PTS / (2 * (FGA + 0.44 * FTA)). |
| TOV% | float | Turnover Percentage: Turnovers per 100 possessions. |
| ORB% | float | Offensive Rebound Percentage. |
| FT/FGA | float | Free Throw Rate: FT attempts per field-goal attempt. |
| 3PAr | float | Three-Point Attempt Rate: 3PA per FGA. |
| Win | int (0/1) | Game outcome flag (1 = win, 0 = loss). |

Notes:
- Both datasets contain these core fields; the model-ready file includes Date and Team/Win explicitly for modeling.
- Fill in any additional engineered features you add in future iterations here.***
