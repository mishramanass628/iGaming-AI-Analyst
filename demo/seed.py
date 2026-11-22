"""Build a local DuckDB file with dummy iGaming data (Jan 2025 – Nov 2026).

Stories baked in so the analysis has something to find:
- Brazil grows fast through 2026; Germany slots stakes drop after the July 2026 stake-limit rule.
- 'Autumn Spins 2026' (October) lifts sign-ups but costs a lot in bonuses.
- Affiliate 7 ('PlayHub Media') brings many sign-ups with poor FTD conversion and heavy bonus use.
- A few self-excluded players still have activity after their exclusion date (compliance breach).

Run directly:  python -m demo.seed
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-11-20")
TOTAL_DAYS = (END - START).days

COUNTRIES = ["GB", "DE", "SE", "CA", "BR", "FI", "IN", "NZ"]
COUNTRY_W = [0.24, 0.17, 0.09, 0.12, 0.18, 0.06, 0.08, 0.06]
CHANNELS = ["Affiliate", "Paid Search", "Social", "Organic", "Referral"]
CHANNEL_W = [0.40, 0.20, 0.15, 0.20, 0.05]
CATEGORIES = ["Slots", "Live Casino", "Table Games", "Sportsbook", "Crash"]
PAY = {"GB": ["Card", "PayPal", "Trustly"], "DE": ["Card", "PayPal", "Trustly", "Skrill"], "SE": ["Trustly", "Card"],
       "CA": ["Interac", "Card"], "BR": ["Pix", "Pix", "Card"], "FI": ["Trustly", "Card"], "IN": ["UPI", "UPI", "Skrill"],
       "NZ": ["Card", "Skrill"]}

AFFILIATES = [
    (1, "Casino Insider", "CPA", 120, 0.0), (2, "BetFinder", "RevShare", 0, 0.30),
    (3, "SlotGuru", "Hybrid", 60, 0.20), (4, "NordicPlay Reviews", "CPA", 150, 0.0),
    (5, "Brasil Apostas Hub", "CPA", 80, 0.0), (6, "TopOdds", "RevShare", 0, 0.35),
    (7, "PlayHub Media", "CPA", 100, 0.0), (8, "CasinoRank", "Hybrid", 50, 0.25),
    (9, "Maple Bets Guide", "CPA", 130, 0.0), (10, "LiveDealer Pro", "RevShare", 0, 0.30),
    (11, "Kiwi Casino Club", "CPA", 110, 0.0), (12, "Cricket & Casino", "Hybrid", 40, 0.20),
]
AFF_W = np.array([8, 7, 8, 5, 9, 5, 22, 7, 6, 6, 4, 13], dtype=float)

CAMPAIGNS = [
    (1, "Welcome Offer 2025", "Acquisition", "2025-01-01", "2025-12-31", 400000),
    (2, "Welcome Offer 2026", "Acquisition", "2026-01-01", "2026-12-31", 450000),
    (3, "Brazil Launch Boost", "Acquisition", "2025-01-01", "2025-03-31", 60000),
    (4, "Summer Cashback 2026", "Retention", "2026-07-01", "2026-08-31", 90000),
    (5, "Autumn Spins 2026", "Reactivation", "2026-10-01", "2026-10-31", 70000),
    (6, "VIP Reload", "VIP", "2025-01-01", "2026-12-31", 250000),
]


def _games(rng):
    providers = {"Slots": ["NetEnt", "Pragmatic Play", "Play'n GO", "Hacksaw"], "Live Casino": ["Evolution", "Pragmatic Live"],
                 "Table Games": ["NetEnt", "In-house"], "Sportsbook": ["In-house Sportsbook"], "Crash": ["Spribe", "In-house"]}
    names = {
        "Slots": ["Golden Pharaoh", "Neon Fruits", "Viking Gold", "Candy Cascade", "Wild Safari", "Dragon Vault",
                  "Lucky Clover", "Space Gems", "Aztec Riches", "Pirate Bounty", "Mystic Moon", "Royal Sevens",
                  "Jungle Spins", "Ice Queen", "Big Bass Bonanza Lite", "Thunder Reels", "Sugar Rush Mini", "Lotus Fortune"],
        "Live Casino": ["Live Roulette", "Lightning Roulette", "Live Blackjack VIP", "Live Blackjack", "Live Baccarat",
                        "Crazy Wheel", "Mega Ball Live", "Live Poker"],
        "Table Games": ["European Roulette", "Classic Blackjack", "Punto Banco", "Casino Hold'em", "Video Poker"],
        "Sportsbook": ["Football", "Tennis", "Basketball", "Cricket", "Ice Hockey", "Esports"],
        "Crash": ["Aviator", "Rocket X", "Plinko Drop"],
    }
    rtp_range = {"Slots": (0.945, 0.97), "Live Casino": (0.973, 0.987), "Table Games": (0.975, 0.99),
                 "Sportsbook": (0.92, 0.94), "Crash": (0.96, 0.975)}
    rows, gid = [], 1
    for cat, ns in names.items():
        for n in ns:
            rows.append((gid, n, rng.choice(providers[cat]), cat, round(rng.uniform(*rtp_range[cat]), 4),
                         (START - pd.Timedelta(days=int(rng.integers(30, 900)))).date()))
            gid += 1
    return pd.DataFrame(rows, columns=["game_id", "game_name", "provider", "category", "rtp", "release_date"])


def build(path: Path, n_players: int = 6000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- players
    N = n_players
    country = rng.choice(COUNTRIES, N, p=COUNTRY_W)
    u = rng.beta(1.3, 1.0, N)
    br = country == "BR"
    u[br] = rng.beta(2.6, 1.0, br.sum())
    signup_off = (u * (TOTAL_DAYS - 5)).astype(int)
    oct_start = (pd.Timestamp("2026-10-01") - START).days
    boost = rng.choice(N, int(N * 0.06), replace=False)
    signup_off[boost] = rng.integers(oct_start, oct_start + 31, len(boost))

    channel = rng.choice(CHANNELS, N, p=CHANNEL_W)
    aff = np.where(channel == "Affiliate", rng.choice(np.arange(1, 13), N, p=AFF_W / AFF_W.sum()), 0)
    segment = rng.choice(["casual", "regular", "high"], N, p=[0.75, 0.20, 0.05])
    segment[(aff == 7) & (rng.random(N) < 0.9)] = "casual"
    depositor = rng.random(N) < np.where(aff == 7, 0.42, 0.80)

    life_mean = {"casual": 45, "regular": 220, "high": 320}
    act_mean = {"casual": 6, "regular": 45, "high": 90}
    stake_mean = {"casual": 25, "regular": 120, "high": 1500}
    bet_size = {"casual": 0.8, "regular": 2.0, "high": 20.0}
    pref_cat = rng.choice(CATEGORIES, N, p=[0.55, 0.2, 0.1, 0.1, 0.05])
    pref_cat[(country == "IN") & (rng.random(N) < 0.6)] = "Sportsbook"

    games = _games(rng)
    by_cat = {c: games.loc[games.category == c, "game_id"].to_numpy() for c in CATEGORIES}
    rtp_by_game = games.set_index("game_id")["rtp"]
    cat_by_game = games.set_index("game_id")["category"]

    # ---------------------------------------------------------------- activity
    p_idx, d_off = [], []
    for i in range(N):
        if not depositor[i]:
            continue
        lifetime = int(rng.exponential(life_mean[segment[i]])) + 1
        avail = max(1, min(lifetime, TOTAL_DAYS - signup_off[i]))
        n = int(np.clip(rng.poisson(act_mean[segment[i]]), 1, avail))
        days = np.sort(rng.choice(avail, n, replace=False)) + signup_off[i]
        days[0] = signup_off[i] + int(rng.integers(0, min(3, avail)))
        p_idx.append(np.full(n, i))
        d_off.append(days)
    p_idx = np.concatenate(p_idx)
    d_off = np.concatenate(d_off)
    # ~25% of days the player also plays a second game
    extra = rng.random(len(p_idx)) < 0.25
    p_idx = np.concatenate([p_idx, p_idx[extra]])
    d_off = np.concatenate([d_off, d_off[extra]])
    R = len(p_idx)

    seg_r = segment[p_idx]
    use_pref = rng.random(R) < 0.7
    cat_r = np.where(use_pref, pref_cat[p_idx], rng.choice(CATEGORIES, R, p=[0.45, 0.2, 0.15, 0.12, 0.08]))
    game_r = np.array([rng.choice(by_cat[c]) for c in cat_r])
    sigma = 0.9
    mean_r = np.vectorize(stake_mean.get)(seg_r).astype(float)
    stake = rng.lognormal(np.log(mean_r) - sigma**2 / 2, sigma)
    dates = START + pd.to_timedelta(d_off, "D")
    stake *= np.where(dates.day_of_week >= 5, 1.25, 1.0)
    de_rule = (country[p_idx] == "DE") & (cat_r == "Slots") & (dates >= pd.Timestamp("2026-07-01"))
    stake *= np.where(de_rule, 0.55, 1.0)
    br_growth = (country[p_idx] == "BR") & (dates >= pd.Timestamp("2026-01-01"))
    stake *= np.where(br_growth, 1.2, 1.0)
    vol = pd.Series(cat_r).map({"Slots": 0.35, "Live Casino": 0.2, "Table Games": 0.25, "Sportsbook": 0.5, "Crash": 0.45}).to_numpy()
    vol = vol * 0.6
    noise = np.exp(rng.normal(-vol**2 / 2, vol))
    rtp = rtp_by_game.loc[game_r].to_numpy()
    win = stake * rtp * noise
    rounds = np.maximum(1, (stake / np.vectorize(bet_size.get)(seg_r)).astype(int))
    if cat_r is not None:
        rounds = np.where(cat_r == "Sportsbook", np.maximum(1, rounds // 15), rounds)

    activity = pd.DataFrame({
        "player_id": p_idx + 1, "game_id": game_r, "activity_date": dates.date,
        "rounds": rounds, "total_stake": stake.round(2), "total_win": win.round(2),
    }).groupby(["player_id", "game_id", "activity_date"], as_index=False).sum()

    # ---------------------------------------------------------------- players table
    lifetime_stake = activity.groupby("player_id")["total_stake"].sum()
    ls = pd.Series(0.0, index=np.arange(1, N + 1)).add(lifetime_stake, fill_value=0).to_numpy()
    vip = np.select([ls >= 250000, ls >= 100000, ls >= 10000, ls >= 1000], ["Diamond", "Platinum", "Gold", "Silver"], "Bronze")
    first_act = activity.groupby("player_id")["activity_date"].min()
    last_act = activity.groupby("player_id")["activity_date"].max()

    self_excl = pd.Series(pd.NaT, index=np.arange(1, N + 1), dtype="datetime64[ns]")
    cand = [pid for pid in last_act.index if segment[pid - 1] != "casual"]
    chosen = rng.choice(cand, 40, replace=False)
    for k, pid in enumerate(chosen):
        fa, la = pd.Timestamp(first_act[pid]), pd.Timestamp(last_act[pid])
        if k < 4 and (la - fa).days > 20:  # breach: activity continues after exclusion
            self_excl[pid] = fa + (la - fa) * 0.6
        else:
            self_excl[pid] = la + pd.Timedelta(days=int(rng.integers(1, 10)))
    self_excl = self_excl.where(self_excl <= END)

    birth_year = rng.integers(1960, 2006, N)
    players = pd.DataFrame({
        "player_id": np.arange(1, N + 1),
        "signup_date": (START + pd.to_timedelta(signup_off, "D")).date,
        "country": country,
        "acquisition_channel": channel,
        "affiliate_id": pd.array(np.where(aff > 0, aff, None), dtype="Int64"),
        "vip_tier": vip,
        "birth_year": birth_year,
        "kyc_status": np.where(depositor, rng.choice(["Verified", "Verified", "Verified", "Pending"], N), "Not started"),
        "self_excluded_at": self_excl.dt.date.to_numpy(),
    })

    # ---------------------------------------------------------------- transactions
    tx = []
    tid = 1
    act_days = activity.groupby("player_id")["activity_date"].apply(lambda s: np.sort(s.unique()))
    for pid, days in act_days.items():
        c = country[pid - 1]
        total = lifetime_stake[pid] / rng.uniform(4, 10)
        n_dep = max(1, rng.poisson(max(1, len(days) / 3)))
        amounts = np.maximum(10, rng.dirichlet(np.ones(n_dep)) * total)
        ddays = np.concatenate([[days[0]], rng.choice(days, n_dep - 1)]) if n_dep > 1 else np.array([days[0]])
        method = rng.choice(PAY[c])
        for a, d in zip(amounts, ddays):
            ts = pd.Timestamp(d) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
            if rng.random() < 0.06:
                tx.append((tid, pid, "Deposit", round(a, 2), method, "Failed", ts)); tid += 1
                ts += pd.Timedelta(minutes=int(rng.integers(1, 30)))
            tx.append((tid, pid, "Deposit", round(a, 2), method, "Completed", ts)); tid += 1
        if rng.random() < 0.35 and len(days) > 1:
            for _ in range(int(rng.integers(1, 4))):
                d = rng.choice(days[1:])
                ts = pd.Timestamp(d) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
                st = rng.choice(["Completed"] * 18 + ["Pending", "Rejected"])
                tx.append((tid, pid, "Withdrawal", round(total * rng.uniform(0.05, 0.3), 2), method, st, ts)); tid += 1
    transactions = pd.DataFrame(tx, columns=["transaction_id", "player_id", "transaction_type", "amount_eur",
                                             "payment_method", "status", "created_at"])

    # ---------------------------------------------------------------- bonuses
    bon = []
    bid = 1
    ftd = transactions[(transactions.transaction_type == "Deposit") & (transactions.status == "Completed")] \
        .sort_values("created_at").groupby("player_id").first()

    def status(issued):
        if issued > END - pd.Timedelta(days=14):
            return rng.choice(["Active", "Converted", "Expired"], p=[0.6, 0.25, 0.15])
        return rng.choice(["Converted", "Expired", "Forfeited"], p=[0.6, 0.33, 0.07])

    for pid, row in ftd.iterrows():
        issued = row.created_at
        if rng.random() < (0.95 if aff[pid - 1] == 7 else 0.7):
            camp = 1 if issued.year == 2025 else 2
            bon.append((bid, pid, camp, "Deposit Match", round(min(row.amount_eur, 100), 2), 35, issued, status(issued))); bid += 1
        if country[pid - 1] == "BR" and issued <= pd.Timestamp("2025-03-31"):
            bon.append((bid, pid, 3, "Free Bet", 50.0, 10, issued, status(issued))); bid += 1

    act = activity.assign(activity_date=pd.to_datetime(activity.activity_date))
    summer = act[(act.activity_date >= "2026-07-01") & (act.activity_date <= "2026-08-31")]
    for pid, g in summer.groupby("player_id"):
        if rng.random() < 0.3:
            loss = max(0.0, g.total_stake.sum() - g.total_win.sum())
            if loss > 5:
                issued = g.activity_date.max() + pd.Timedelta(days=1)
                bon.append((bid, pid, 4, "Cashback", round(loss * 0.1, 2), 1, issued, status(issued))); bid += 1
    octo = act[(act.activity_date >= "2026-10-01") & (act.activity_date <= "2026-10-31")]
    for pid, g in octo.groupby("player_id"):
        p = 0.95 if aff[pid - 1] == 7 else 0.6
        if rng.random() < p:
            issued = g.activity_date.min()
            amount = rng.uniform(25, 60) if aff[pid - 1] == 7 else rng.uniform(10, 50)
            bon.append((bid, pid, 5, "Free Spins", round(amount, 2), 25, issued, status(issued))); bid += 1
    act["month"] = act.activity_date.dt.to_period("M")
    vip_players = set(players.loc[players.vip_tier.isin(["Gold", "Platinum", "Diamond"]), "player_id"])
    monthly = act[act.player_id.isin(vip_players)].groupby(["player_id", "month"]).total_stake.sum().reset_index()
    for r in monthly.itertuples():
        if rng.random() < 0.5:
            issued = r.month.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
            bon.append((bid, r.player_id, 6, "Reload", round(r.total_stake * 0.005, 2), 20, issued, status(issued))); bid += 1
    bonuses = pd.DataFrame(bon, columns=["bonus_id", "player_id", "campaign_id", "bonus_type", "amount_eur",
                                         "wagering_multiplier", "issued_at", "status"])

    campaigns = pd.DataFrame(CAMPAIGNS, columns=["campaign_id", "campaign_name", "objective", "start_date", "end_date", "budget_eur"])
    campaigns["start_date"] = pd.to_datetime(campaigns.start_date).dt.date
    campaigns["end_date"] = pd.to_datetime(campaigns.end_date).dt.date
    affiliates = pd.DataFrame(AFFILIATES, columns=["affiliate_id", "affiliate_name", "commission_model", "cpa_eur", "revshare_pct"])

    # ---------------------------------------------------------------- write
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    for name, df in [("players", players), ("games", games), ("daily_game_activity", activity),
                     ("transactions", transactions), ("bonuses", bonuses), ("campaigns", campaigns),
                     ("affiliates", affiliates)]:
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    comments = {
        "players": "One row per registered player. A player without a completed deposit never made an FTD.",
        "games": "Game catalogue. category: Slots, Live Casino, Table Games, Sportsbook, Crash. rtp = theoretical return to player.",
        "daily_game_activity": "Player x game x day aggregate of wagering. GGR = total_stake - total_win.",
        "transactions": "Deposits and withdrawals in EUR. Only status='Completed' moves money.",
        "bonuses": "Bonuses granted to players. Bonus cost counts status IN ('Converted','Active').",
        "campaigns": "Marketing campaigns that bonuses belong to.",
        "affiliates": "Affiliate partners. players.affiliate_id references this table.",
    }
    for t, c in comments.items():
        con.execute(f"COMMENT ON TABLE {t} IS '{c.replace(chr(39), chr(39)*2)}'")
    col_comments = {
        ("players", "self_excluded_at"): "Date the player self-excluded; NULL if never. No activity is allowed after it.",
        ("players", "vip_tier"): "Bronze, Silver, Gold, Platinum, Diamond (by lifetime stake)",
        ("daily_game_activity", "total_stake"): "Amount wagered in EUR (turnover)",
        ("daily_game_activity", "total_win"): "Amount paid back to the player in EUR",
        ("transactions", "transaction_type"): "'Deposit' or 'Withdrawal'",
        ("transactions", "status"): "'Completed', 'Failed', 'Pending' or 'Rejected'",
        ("bonuses", "status"): "'Converted', 'Active', 'Expired' or 'Forfeited'",
    }
    for (t, col), c in col_comments.items():
        con.execute(f"COMMENT ON COLUMN {t}.{col} IS '{c.replace(chr(39), chr(39)*2)}'")
    con.close()
    return path


if __name__ == "__main__":
    out = build(Path(__file__).parent / "igaming_demo.duckdb")
    con = duckdb.connect(str(out), read_only=True)
    for t in ["players", "games", "daily_game_activity", "transactions", "bonuses", "campaigns", "affiliates"]:
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
