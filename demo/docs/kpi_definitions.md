# KPI Definitions Handbook (Finance & BI) — v3.2, 2026

All money is in EUR. Dates use the player's activity date (UTC).

## Revenue
- **Turnover (Stake)**: SUM(daily_game_activity.total_stake).
- **GGR (Gross Gaming Revenue)**: SUM(total_stake) - SUM(total_win). Always computed from daily_game_activity.
- **Hold %**: GGR / Turnover. Expected range: Slots 3–5.5%, Live Casino 1.3–2.7%, Table Games 1–2.5%, Sportsbook 6–8%, Crash 2.5–4%.
- **Bonus cost**: SUM(bonuses.amount_eur) WHERE status IN ('Converted','Active'). Expired and Forfeited bonuses are NOT a cost.
- **Gaming duty**: GGR × country duty rate. Rates: GB 21%, DE 5.3%, SE 22%, CA 20%, BR 12%, FI 0% (offshore licence), IN 28%, NZ 0%.
- **NGR (Net Gaming Revenue)**: GGR - bonus cost - gaming duty. NGR is the headline revenue KPI for leadership.

## Players
- **Registration**: a row in players (signup_date).
- **FTD (First-Time Depositor)**: a player's first transaction with transaction_type = 'Deposit' AND status = 'Completed'. The FTD date is that transaction's date.
- **FTD conversion rate**: FTDs / registrations in the same cohort (by signup month).
- **Active player**: a player with at least one row in daily_game_activity in the period.
- **ARPU**: NGR / active players in the period. GGR-based ARPU must be labelled "GGR ARPU".
- **Churned player**: no activity for 30+ consecutive days as of the reporting date.
- **Retention (M1)**: share of a signup-month cohort that is active in the following calendar month.
- **Net deposits**: completed deposits - completed withdrawals.

## Bonus efficiency
- **Bonus-to-GGR ratio**: bonus cost / GGR. Healthy: below 20%. Above 30% triggers a review by CRM.
- **Bonus abuse signal**: players whose bonus cost exceeds 50% of their completed deposits, or who claim a bonus and never deposit again.
