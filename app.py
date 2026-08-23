"""
FUTBOL ANALİZ - Backend (Sofascore Proxy + Poisson Motoru)
Sofascore unofficial API üzerinden veri çeker, Poisson modeli hesaplar.
"""

import os, time, math, threading, json, logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Sofascore Headers ─────────────────────────────────────────────────────────
SS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Origin": "https://www.sofascore.com",
}
BASE = "https://api.sofascore.com/api/v1"

# ─── In-Memory Cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_lock = threading.Lock()

def _cache_get(key):
    with _lock:
        e = _cache.get(key)
        if e and time.time() < e["exp"]:
            return e["data"]
    return None

def _cache_set(key, data, ttl=3600):
    with _lock:
        _cache[key] = {"data": data, "exp": time.time() + ttl}

def ss(path, params=None, ttl=3600):
    """Sofascore'a GET isteği at, cache'le."""
    key = path + str(sorted((params or {}).items()))
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(f"{BASE}{path}", headers=SS_HEADERS,
                         params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        _cache_set(key, data, ttl)
        return data
    except Exception as exc:
        log.warning("Sofascore hata %s: %s", path, exc)
        return None

# ─── Poisson Motoru ────────────────────────────────────────────────────────────
def _pois(lam: float, k: int) -> float:
    """Poisson olasılığı P(X=k | λ)"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_matrix(home_xg: float, away_xg: float, max_g: int = 8):
    """Tam skor matrisi ve toplu istatistikler."""
    mat = [[_pois(home_xg, i) * _pois(away_xg, j)
            for j in range(max_g + 1)]
           for i in range(max_g + 1)]

    hw = sum(mat[i][j] for i in range(max_g+1) for j in range(max_g+1) if i > j)
    dr = sum(mat[i][i] for i in range(max_g+1))
    aw = 1.0 - hw - dr

    ov25 = sum(mat[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j > 2)
    btts = sum(mat[i][j] for i in range(1, max_g+1) for j in range(1, max_g+1))
    ov15 = sum(mat[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j > 1)
    ov35 = sum(mat[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j > 3)
    ov45 = sum(mat[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j > 4)

    scores = sorted(
        [{"home": i, "away": j, "prob": round(mat[i][j] * 100, 2)}
         for i in range(max_g+1) for j in range(max_g+1)],
        key=lambda x: -x["prob"]
    )[:16]

    # 5x5 görsel matris
    visual = [[round(mat[i][j]*100, 1) for j in range(5)] for i in range(5)]

    return {
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "home_win": round(hw * 100, 2),
        "draw":     round(dr * 100, 2),
        "away_win": round(aw * 100, 2),
        "over15":   round(ov15 * 100, 2),
        "under15":  round((1 - ov15) * 100, 2),
        "over25":   round(ov25 * 100, 2),
        "under25":  round((1 - ov25) * 100, 2),
        "over35":   round(ov35 * 100, 2),
        "under35":  round((1 - ov35) * 100, 2),
        "over45":   round(ov45 * 100, 2),
        "under45":  round((1 - ov45) * 100, 2),
        "btts":     round(btts * 100, 2),
        "no_btts":  round((1 - btts) * 100, 2),
        "top_scores": scores,
        "matrix_5x5": visual,
    }

# ─── Yardımcı: Maç listesinden istatistik çıkar ───────────────────────────────
def _extract_team_stats(events: list, team_id: int, venue: str):
    """
    venue = 'home' | 'away' | 'all'
    Son 5 yıl (1825 gün) verisi ağırlıklı ortalama ile hesaplanır.
    Daha yakın maçlar daha fazla ağırlık taşır.
    """
    cutoff = datetime.now() - timedelta(days=5 * 365)
    now_ts = time.time()

    goals_scored, goals_conceded = [], []
    results = []
    goal_times = {p: 0 for p in ["0-15","16-30","31-45","46-60","61-75","76-90","90+"]}

    for ev in events:
        ts = ev.get("startTimestamp", 0)
        if ts < cutoff.timestamp():
            continue

        home_team = ev.get("homeTeam", {})
        away_team = ev.get("awayTeam", {})
        home_id   = home_team.get("id")
        away_id   = away_team.get("id")

        is_home = (home_id == team_id)
        is_away = (away_id == team_id)

        if venue == "home" and not is_home: continue
        if venue == "away" and not is_away: continue
        if not (is_home or is_away): continue

        score = ev.get("homeScore", {})
        h_goals = score.get("current", score.get("display", None))
        a_score = ev.get("awayScore", {})
        a_goals = a_score.get("current", a_score.get("display", None))

        if h_goals is None or a_goals is None:
            continue

        h_goals = int(h_goals)
        a_goals = int(a_goals)

        # Ağırlık: son 6 ay 2x, 6-18 ay 1.5x, 18-36 ay 1.2x, daha eski 1x
        age_days = (now_ts - ts) / 86400
        if age_days < 180:   w = 2.0
        elif age_days < 540: w = 1.5
        elif age_days < 1080: w = 1.2
        else:                 w = 1.0

        if is_home:
            for _ in range(round(w * 10)):
                goals_scored.append(h_goals)
                goals_conceded.append(a_goals)
            if h_goals > a_goals: results.append("W")
            elif h_goals == a_goals: results.append("D")
            else: results.append("L")
        else:
            for _ in range(round(w * 10)):
                goals_scored.append(a_goals)
                goals_conceded.append(h_goals)
            if a_goals > h_goals: results.append("W")
            elif a_goals == h_goals: results.append("D")
            else: results.append("L")

    if not goals_scored:
        return None

    avg_scored    = sum(goals_scored)    / len(goals_scored)
    avg_conceded  = sum(goals_conceded)  / len(goals_conceded)

    return {
        "matches":       len(results),
        "avg_scored":    round(avg_scored, 3),
        "avg_conceded":  round(avg_conceded, 3),
        "results":       results[:30],
        "goal_times":    goal_times,
        "wins":   results.count("W"),
        "draws":  results.count("D"),
        "losses": results.count("L"),
    }

def _fetch_team_events(team_id: int, pages: int = 5) -> list:
    """Son ~50 maçı çek (5 sayfa × 10 maç)."""
    all_events = []
    for page in range(pages):
        data = ss(f"/team/{team_id}/events/last/{page}", ttl=1800)
        if not data:
            break
        events = data.get("events", [])
        all_events.extend(events)
        if not data.get("hasNextPage", True) or not events:
            break
    return all_events

def _fetch_h2h(home_id: int, away_id: int) -> list:
    data = ss(f"/event/{home_id}/h2h", ttl=7200)
    if not data:
        # Sofascore H2H endpoint team-based
        data = ss(f"/team/{home_id}/h2h-events/{away_id}", ttl=7200)
    if not data:
        return []
    return data.get("events", data.get("previousEvents", []))[:20]

# ─── API Endpoint'leri ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    data = ss(f"/search/multi-search", {"q": q})
    if not data:
        return jsonify([])
    teams = []
    for item in data.get("results", []):
        if item.get("type") == "team" and item.get("entity", {}).get("sport", {}).get("slug") == "football":
            e = item["entity"]
            teams.append({
                "id": e["id"],
                "name": e["name"],
                "shortName": e.get("shortName", e["name"]),
                "country": e.get("country", {}).get("name", ""),
                "logo": f"https://api.sofascore.com/api/v1/team/{e['id']}/image",
            })
        if len(teams) >= 8:
            break
    return jsonify(teams)

@app.route("/api/analyze")
def analyze():
    home_id = request.args.get("home_id", type=int)
    away_id = request.args.get("away_id", type=int)
    home_odds = request.args.get("home_odds", type=float)
    draw_odds = request.args.get("draw_odds", type=float)
    away_odds = request.args.get("away_odds", type=float)

    if not home_id or not away_id:
        return jsonify({"error": "home_id ve away_id gerekli"}), 400

    # Paralel veri çekimi
    home_events = _fetch_team_events(home_id, pages=5)
    away_events = _fetch_team_events(away_id, pages=5)

    # H2H
    h2h_events = []
    h2h_data = ss(f"/team/{home_id}/h2h/{away_id}", ttl=7200)
    if h2h_data:
        h2h_events = h2h_data.get("events", [])[:10]

    # Takım bilgileri
    home_info_raw = ss(f"/team/{home_id}")
    away_info_raw = ss(f"/team/{away_id}")

    def team_info(raw, tid):
        t = (raw or {}).get("team", {})
        return {
            "id": tid,
            "name": t.get("name", f"Takım {tid}"),
            "shortName": t.get("shortName", f"Takım {tid}"),
            "logo": f"https://api.sofascore.com/api/v1/team/{tid}/image",
            "country": t.get("country", {}).get("name", ""),
        }

    home_info = team_info(home_info_raw, home_id)
    away_info = team_info(away_info_raw, away_id)

    # İstatistik hesapla
    h_home = _extract_team_stats(home_events, home_id, "home") or {}
    h_all  = _extract_team_stats(home_events, home_id, "all")  or {}
    a_away = _extract_team_stats(away_events, away_id, "away") or {}
    a_all  = _extract_team_stats(away_events, away_id, "all")  or {}

    # xG hesabı (ağırlıklı: ev+genel)
    h_scored   = h_home.get("avg_scored", h_all.get("avg_scored", 1.3))
    h_conceded = h_home.get("avg_conceded", h_all.get("avg_conceded", 1.1))
    a_scored   = a_away.get("avg_scored", a_all.get("avg_scored", 1.1))
    a_conceded = a_away.get("avg_conceded", a_all.get("avg_conceded", 1.3))

    # Poisson beklenen goller
    # home_xg = ev sahibi atma × deplasman savunma zayıflığı
    # Normalize: lig ortalaması varsayılan 1.4 / 1.1
    HOME_AVG = 1.4
    AWAY_AVG = 1.1
    home_xg = h_scored   * (a_conceded / AWAY_AVG)
    away_xg = a_scored   * (h_conceded / HOME_AVG)
    home_xg = max(0.3, min(home_xg, 5.0))
    away_xg = max(0.3, min(away_xg, 5.0))

    probs = poisson_matrix(home_xg, away_xg)

    # Oran analizi
    def value_analysis(model_pct, odds_val, label):
        if not odds_val or odds_val <= 1.0:
            return None
        implied = round(100 / odds_val, 2)
        edge    = round(model_pct - implied, 2)
        if edge > 5:
            verdict = "▲ Değerli"
            cls = "value"
        elif edge < -5:
            verdict = "▼ Pahalı"
            cls = "overpriced"
        else:
            verdict = "≈ Adil"
            cls = "fair"
        return {
            "label": label,
            "odds": odds_val,
            "implied_pct": implied,
            "model_pct": model_pct,
            "edge": edge,
            "verdict": verdict,
            "cls": cls,
        }

    odds_analysis = []
    if home_odds:
        v = value_analysis(probs["home_win"], home_odds, "Ev Sahibi Kazanır")
        if v: odds_analysis.append(v)
    if draw_odds:
        v = value_analysis(probs["draw"], draw_odds, "Beraberlik")
        if v: odds_analysis.append(v)
    if away_odds:
        v = value_analysis(probs["away_win"], away_odds, "Deplasman Kazanır")
        if v: odds_analysis.append(v)

    # H2H özet
    h2h_summary = []
    for ev in h2h_events[:10]:
        hs = ev.get("homeScore", {}).get("current")
        as_ = ev.get("awayScore", {}).get("current")
        if hs is None: continue
        ht = ev.get("homeTeam", {})
        at = ev.get("awayTeam", {})
        h2h_summary.append({
            "date": datetime.fromtimestamp(ev.get("startTimestamp",0)).strftime("%d.%m.%Y"),
            "home": ht.get("shortName", ht.get("name", "?")),
            "away": at.get("shortName", at.get("name", "?")),
            "score": f"{hs} - {as_}",
            "home_id": ht.get("id"),
            "away_id": at.get("id"),
        })

    # Son maçlar özeti
    def recent_matches(events, team_id, n=10):
        matches = []
        for ev in events:
            hs = ev.get("homeScore", {}).get("current")
            as_ = ev.get("awayScore", {}).get("current")
            if hs is None: continue
            ht = ev.get("homeTeam", {})
            at = ev.get("awayTeam", {})
            is_home = ht.get("id") == team_id
            if is_home:
                res = "W" if hs > as_ else ("D" if hs == as_ else "L")
            else:
                res = "W" if as_ > hs else ("D" if as_ == hs else "L")
            matches.append({
                "date": datetime.fromtimestamp(ev.get("startTimestamp",0)).strftime("%d.%m.%Y"),
                "home": ht.get("shortName", ht.get("name","?")),
                "away": at.get("shortName", at.get("name","?")),
                "score": f"{hs}-{as_}",
                "result": res,
                "is_home": is_home,
            })
            if len(matches) >= n: break
        return matches

    return jsonify({
        "home": home_info,
        "away": away_info,
        "poisson": probs,
        "home_stats": {
            "home": h_home,
            "all": h_all,
        },
        "away_stats": {
            "away": a_away,
            "all": a_all,
        },
        "odds_analysis": odds_analysis,
        "h2h": h2h_summary,
        "home_recent": recent_matches(home_events, home_id),
        "away_recent": recent_matches(away_events, away_id),
    })

@app.route("/api/upcoming/<int:league_id>")
def upcoming(league_id):
    data = ss(f"/unique-tournament/{league_id}/scheduled-events", ttl=900)
    if not data:
        return jsonify([])
    events = []
    for ev in (data.get("events") or [])[:30]:
        hs = ev.get("homeScore", {})
        as_ = ev.get("awayScore", {})
        events.append({
            "id": ev.get("id"),
            "date": datetime.fromtimestamp(ev.get("startTimestamp",0)).strftime("%d.%m.%Y %H:%M"),
            "home": ev.get("homeTeam", {}).get("name","?"),
            "home_id": ev.get("homeTeam", {}).get("id"),
            "away": ev.get("awayTeam", {}).get("name","?"),
            "away_id": ev.get("awayTeam", {}).get("id"),
            "home_logo": f"https://api.sofascore.com/api/v1/team/{ev.get('homeTeam',{}).get('id','')}/image",
            "away_logo": f"https://api.sofascore.com/api/v1/team/{ev.get('awayTeam',{}).get('id','')}/image",
            "status": ev.get("status", {}).get("description",""),
            "score": f"{hs.get('current','-')} - {as_.get('current','-')}" if hs.get("current") is not None else None,
        })
    return jsonify(events)

@app.route("/api/live")
def live():
    data = ss("/sport/football/events/live", ttl=60)
    if not data:
        return jsonify([])
    events = []
    for ev in (data.get("events") or [])[:50]:
        hs = ev.get("homeScore", {})
        as_ = ev.get("awayScore", {})
        events.append({
            "id": ev.get("id"),
            "home": ev.get("homeTeam", {}).get("name","?"),
            "home_id": ev.get("homeTeam", {}).get("id"),
            "away": ev.get("awayTeam", {}).get("name","?"),
            "away_id": ev.get("awayTeam", {}).get("id"),
            "home_score": hs.get("current",0),
            "away_score": as_.get("current",0),
            "minute": ev.get("time", {}).get("played", "?"),
            "tournament": ev.get("tournament", {}).get("name",""),
            "home_logo": f"https://api.sofascore.com/api/v1/team/{ev.get('homeTeam',{}).get('id','')}/image",
            "away_logo": f"https://api.sofascore.com/api/v1/team/{ev.get('awayTeam',{}).get('id','')}/image",
        })
    return jsonify(events)

@app.route("/api/odds/<int:event_id>")
def odds(event_id):
    data = ss(f"/event/{event_id}/odds/1/all", ttl=300)
    if not data:
        return jsonify({})
    markets = {}
    for mkt in (data.get("markets") or []):
        name = mkt.get("marketName","")
        choices = []
        for ch in mkt.get("choices", []):
            choices.append({
                "name": ch.get("name"),
                "odds": ch.get("fractionalValue") or ch.get("odds"),
            })
        markets[name] = choices
    return jsonify(markets)

@app.route("/api/standings/<int:league_id>/<int:season_id>")
def standings(league_id, season_id):
    data = ss(f"/unique-tournament/{league_id}/season/{season_id}/standings/total", ttl=3600)
    if not data:
        return jsonify([])
    rows = []
    for st in (data.get("standings") or [{}])[0].get("rows", []):
        t = st.get("team", {})
        rows.append({
            "pos": st.get("position"),
            "team": t.get("name","?"),
            "team_id": t.get("id"),
            "played": st.get("matches"),
            "w": st.get("wins"),
            "d": st.get("draws"),
            "l": st.get("losses"),
            "gf": st.get("scoresFor"),
            "ga": st.get("scoresAgainst"),
            "pts": st.get("points"),
        })
    return jsonify(rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
