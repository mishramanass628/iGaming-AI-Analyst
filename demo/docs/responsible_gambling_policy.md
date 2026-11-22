# Responsible Gambling & Compliance Policy — 2026

## Self-exclusion
- When a player self-excludes (players.self_excluded_at), the account must be closed immediately. **Any wagering, deposit or bonus after self_excluded_at is a regulatory breach** and must be reported to Compliance within 24 hours. In GB this is reportable to the regulator.

## Markers of harm (automated monitoring)
Flag a player for review when any of these happen in a calendar month:
1. More than 20 completed deposits.
2. Completed deposits above €5,000 (GB: €2,000).
3. Three or more failed deposits in one day.
4. Net losses (stake - win) above €10,000.
5. Players aged under 25 (by birth_year) with deposits above €1,000.

## Affordability
- Platinum and Diamond players require documented source-of-funds checks.
- KYC: players with kyc_status <> 'Verified' must not withdraw. Deposits above €2,000 lifetime require Verified KYC.

## Bonuses
- Do not send bonus offers to players flagged under markers of harm or self-excluded.
