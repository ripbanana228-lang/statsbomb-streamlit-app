import os, re, ast, json, io, zipfile, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mplsoccer import Pitch

import streamlit as st
from statsbombpy import sb  # StatsBomb Open Data endpoints

# ---------- App config ----------
st.set_page_config(page_title="StatsBomb Open Data Match Report", layout="wide")

# Matplotlib defaults (readable in browser)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 11

PITCH_L = 120
PITCH_W = 80


# ---------- Utilities (ported from notebook) ----------
# 1) Imports y configuración
import os, re, ast, json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mplsoccer import Pitch



# Ajustes generales de plots
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 11



def slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def ensure_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
        try:
            vv = ast.literal_eval(v)
            if isinstance(vv, (list, tuple)):
                return list(vv)
        except Exception:
            return None
    return None

def split_xy(df: pd.DataFrame, col: str, xcol: str, ycol: str) -> pd.DataFrame:
    if col not in df.columns:
        if xcol not in df.columns: df[xcol] = np.nan
        if ycol not in df.columns: df[ycol] = np.nan
        return df
    v = df[col].apply(ensure_list)
    df[xcol] = v.apply(lambda z: z[0] if isinstance(z, list) and len(z) >= 2 else np.nan)
    df[ycol] = v.apply(lambda z: z[1] if isinstance(z, list) and len(z) >= 2 else np.nan)
    return df

def detect_team_col(ev: pd.DataFrame) -> str:
    for c in ["team", "team_name"]:
        if c in ev.columns:
            return c
    raise ValueError("No encuentro columna de equipo (team/team_name) en events.")

def detect_player_col(df: pd.DataFrame) -> str:
    for c in ["player", "player_name"]:
        if c in df.columns:
            return c
    raise ValueError("No encuentro columna de jugador (player/player_name).")

def detect_xg_col(shots: pd.DataFrame) -> str:
    for c in ["shot_statsbomb_xg", "shot_xg", "statsbomb_xg"]:
        if c in shots.columns:
            return c
    return ""

def detect_goal_mask(shots: pd.DataFrame) -> pd.Series:
    for c in ["shot_outcome", "shot_outcome_name"]:
        if c in shots.columns:
            s = shots[c].astype(str).str.lower()
            return s.str.contains("goal")
    if "outcome" in shots.columns:
        s = shots["outcome"].astype(str).str.lower()
        return s.str.contains("goal")
    return pd.Series(False, index=shots.index)

def open_play_completed_passes(passes: pd.DataFrame) -> pd.DataFrame:
    df = passes.copy()
    # completados
    if "pass_outcome" in df.columns:
        df = df[df["pass_outcome"].isna()].copy()
    # quitar ABP
    for c in ["pass_type", "pass_type_name"]:
        if c in df.columns:
            t = df[c].astype(str).str.lower()
            df = df[~t.isin(["corner", "free kick", "throw-in", "goal kick", "kick off"])].copy()
            break
    return df

def first_half_window(df: pd.DataFrame, max_min: int = 15) -> pd.DataFrame:
    out = df.copy()
    if "period" in out.columns:
        p = pd.to_numeric(out["period"], errors="coerce").fillna(1).astype(int)
        out = out[p == 1].copy()
    if "minute" in out.columns:
        m = pd.to_numeric(out["minute"], errors="coerce").fillna(0).astype(int)
        out = out[m <= max_min].copy()
    return out


# ---------- Data layer (Streamlit cached) ----------
@st.cache_data(show_spinner=False)
def get_competitions() -> pd.DataFrame:
    return sb.competitions().copy()

@st.cache_data(show_spinner=False)
def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    return sb.matches(competition_id=int(competition_id), season_id=int(season_id)).copy()

@st.cache_data(show_spinner=False)
def get_events(match_id: int) -> pd.DataFrame:
    return sb.events(match_id=int(match_id)).copy()

@st.cache_data(show_spinner=False)
def get_lineups(match_id: int) -> pd.DataFrame:
    df = sb.lineups(match_id=int(match_id))

    # Case A: DataFrame
    if isinstance(df, pd.DataFrame):
        return df.copy()

    rows = []
    # Case B: dict {team: [players...]} (common)
    if isinstance(df, dict):
        if any(isinstance(v, list) for v in df.values()):
            for team_name, players in df.items():
                if not isinstance(players, list):
                    continue
                for p in players:
                    row = {"team": team_name}
                    if isinstance(p, dict):
                        row.update(p)
                    else:
                        row["player"] = p
                    rows.append(row)
            return pd.DataFrame(rows)

        # Case C: dict scalars
        return pd.DataFrame([df])

    # Case D: list
    if isinstance(df, list):
        return pd.DataFrame(df)

    # Fallback
    return pd.DataFrame([{"raw": str(df)}])



# ---------- Match context + saving ----------
def match_context(matches_df: pd.DataFrame, match_id: int) -> dict:
    ctx = {"match_id": int(match_id), "home_team": None, "away_team": None,
           "home_score": None, "away_score": None, "match_date": None,
           "competition": None, "season": None}
    if matches_df is None or matches_df.empty:
        return ctx
    if "match_id" not in matches_df.columns:
        return ctx
    row = matches_df.loc[matches_df["match_id"] == int(match_id)]
    if row.empty:
        return ctx
    r = row.iloc[0]

    def pick(cols):
        for c in cols:
            if c in matches_df.columns:
                return r.get(c)
        return None

    ctx["home_team"] = pick(["home_team_name", "home_team"])
    ctx["away_team"] = pick(["away_team_name", "away_team"])
    ctx["home_score"] = pick(["home_score"])
    ctx["away_score"] = pick(["away_score"])
    ctx["match_date"] = pick(["match_date", "date", "kick_off"])
    ctx["competition"] = pick(["competition_name", "competition"])
    ctx["season"] = pick(["season_name", "season"])
    return ctx

def fmt_ctx(ctx: dict) -> str:
    ht = ctx.get("home_team") or "Home"
    at = ctx.get("away_team") or "Away"
    hs, a_s = ctx.get("home_score"), ctx.get("away_score")
    score = f"{hs}-{a_s}" if (hs is not None and a_s is not None) else ""
    meta = " | ".join([str(x) for x in [ctx.get("competition"), ctx.get("season"), ctx.get("match_date")] if x not in [None, "nan", "NaT"] and str(x).strip()])
    head = f"{ht} vs {at}"
    if score:
        head += f" ({score})"
    if meta:
        head += f" — {meta}"
    return head

def match_outdir(ctx: dict) -> Path:
    ht = slug(ctx.get("home_team") or "home")
    at = slug(ctx.get("away_team") or "away")
    mid = int(ctx.get("match_id") or 0)
    d = OUTROOT / f"{ht}_{at}_{mid}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_df(df: pd.DataFrame, outdir: Path, stem: str, save_csv=True, save_parquet=True):
    """
    Guarda DataFrames a CSV y/o Parquet de forma robusta.
    StatsBomb (especialmente lineups) puede traer columnas con objetos complejos
    (listas/dicts/Series/DataFrames). Para Parquet (pyarrow) convertimos esos objetos
    a JSON string de forma segura.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}

    def _to_jsonable(x):
        # Convierte a algo serializable (dict/list/str/None/number)
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, pd.DataFrame):
            return x.to_dict(orient="records")
        if isinstance(x, pd.Series):
            return x.to_list()
        # numpy types
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                pass
        return x

    def _safe_json_str(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return json.dumps(_to_jsonable(x), ensure_ascii=False, default=str)
        except Exception:
            return str(x)

    if save_parquet:
        pq = outdir / f"{stem}.parquet"
        df2 = df.copy()

        # pyarrow falla con dtype object + valores no escalares.
        for c in df2.columns:
            if df2[c].dtype == "object":
                # si hay cualquier valor "complejo", serializa toda la columna
                complex_mask = df2[c].apply(lambda v: isinstance(v, (dict, list, pd.DataFrame, pd.Series)))
                if complex_mask.any():
                    df2[c] = df2[c].apply(_safe_json_str)

        df2.to_parquet(pq, index=False)
        out["parquet"] = pq

    if save_csv:
        csv = outdir / f"{stem}.csv"
        df.to_csv(csv, index=False, encoding="utf-8")
        out["csv"] = csv

    return out



# ---------- Plotting (ported from notebook; modified to return fig) ----------
def prep_events_for_plots(events_df: pd.DataFrame) -> pd.DataFrame:
    ev = events_df.copy()
    ev = split_xy(ev, "location", "x", "y")
    ev = split_xy(ev, "pass_end_location", "pass_end_x", "pass_end_y")
    return ev

def mirror_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a,b in [("x","y"), ("pass_end_x","pass_end_y")]:
        if a in out.columns:
            out[a] = PITCH_L - pd.to_numeric(out[a], errors="coerce")
        if b in out.columns:
            out[b] = PITCH_W - pd.to_numeric(out[b], errors="coerce")
    return out

def normalize_attack_right(df: pd.DataFrame, team: str, ctx: dict) -> pd.DataFrame:
    away = str(ctx.get("away_team") or "")
    if away and str(team) == away:
        return mirror_coords(df)
    return df

def plot_shot_map(events_xy: pd.DataFrame, ctx: dict, save=False, outdir: Optional[Path] = None):
    """
    Shot map (ambos equipos) con:
    - Tamaño ~ xG
    - Estrella = gol
    - Disparos con/ sin gol diferenciados por marcador (círculo vs estrella)
    - Orientación consistente: ambos equipos atacan hacia la derecha (visitante espejado)
    """
    team_col = detect_team_col(events_xy)
    shots = events_xy[events_xy["type"] == "Shot"].copy()
    if shots.empty:
        return

    xg_col = detect_xg_col(shots)
    if not xg_col:
        shots["xg"] = 0.0
        xg_col = "xg"
    shots[xg_col] = pd.to_numeric(shots[xg_col], errors="coerce").fillna(0.0).clip(0, 1)

    # contexto
    teams = [ctx.get("home_team"), ctx.get("away_team")]
    teams = [t for t in teams if t is not None]

    # Colores por equipo (relleno). Si no coincide, usa gris.
    colors = {}
    if len(teams) >= 1:
        colors[str(teams[0])] = "#1f77b4"
    if len(teams) >= 2:
        colors[str(teams[1])] = "#ff7f0e"

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11.5, 7.2))

    # Nota: evitamos colores explícitos; usamos estilos (edge/alpha) y labels.
    legend_handles = []
    legend_labels = []

    for t in teams:
        team_color = colors.get(str(t), "#808080")
        s = shots[shots[team_col] == t].dropna(subset=["x", "y"]).copy()
        if s.empty:
            continue
        s = normalize_attack_right(s, team=t, ctx=ctx)

        # Tamaño en función del xG (más legible que lineal puro)
        sizes = 35 + (s[xg_col] ** 0.55) * 950

        goals = detect_goal_mask(s)
        nong = s[~goals]
        g = s[goals]

        # No-gol: círculos con borde
        sc = pitch.scatter(
            nong["x"], nong["y"],
            s=sizes.loc[nong.index],
            ax=ax, alpha=0.75, linewidth=1.2,
            edgecolor="black", facecolor=team_color,
            label=str(t),
            zorder=3
        )

        # Gol: estrella sólida encima
        if not g.empty:
            pitch.scatter(
                g["x"], g["y"],
                s=240,
                marker="*",
                ax=ax,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.8,
                zorder=4
            )

        legend_handles.append(sc)
        legend_labels.append(f"{t} (xG={s[xg_col].sum():.2f}, tiros={len(s)})")

    # Leyenda xG (tamaños de referencia)
    ref_xg = [0.05, 0.15, 0.30, 0.50]
    ref_sizes = 35 + (np.array(ref_xg) ** 0.55) * 950
    for xg, ss in zip(ref_xg, ref_sizes):
        ax.scatter([], [], s=ss, facecolors="none", edgecolors="black", linewidths=1.2, label=f"xG {xg:.2f}")
    ax.legend(loc="upper right", frameon=True)

    ax.set_title(f"Mapa de tiros (tamaño ~ xG, ★ = gol) — {fmt_ctx(ctx)}", pad=14)

    # Pie de figura: recordatorio de orientación
    ax.text(2, 78.5, "Orientación: ambos equipos atacan →", ha="left", va="top", fontsize=9, alpha=0.75)

    fig.tight_layout()
    if save and outdir is not None:
        (Path(outdir) / "01_shot_map.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outdir) / "01_shot_map.png", bbox_inches="tight", dpi=180)
    return fig


def plot_xg_flow(events_xy: pd.DataFrame, ctx: dict, save=False, outdir: Optional[Path] = None):
    """
    xG acumulado por minuto (step chart) con:
    - Línea para cada equipo
    - Marcadores de gol sobre la curva (si se puede detectar)
    - Halftime / Fulltime como referencia
    """
    team_col = detect_team_col(events_xy)
    shots = events_xy[events_xy["type"] == "Shot"].copy()
    if shots.empty:
        return

    xg_col = detect_xg_col(shots)
    if not xg_col:
        shots["xg"] = 0.0
        xg_col = "xg"

    # tiempo (min + seg) para ordenamiento más fino
    shots["minute"] = pd.to_numeric(shots.get("minute", 0), errors="coerce").fillna(0).astype(int)
    shots["second"] = pd.to_numeric(shots.get("second", 0), errors="coerce").fillna(0).astype(int)
    shots["t"] = shots["minute"] + shots["second"] / 60.0

    shots["xg"] = pd.to_numeric(shots[xg_col], errors="coerce").fillna(0.0).clip(0, 1)

    teams = [ctx.get("home_team"), ctx.get("away_team")]
    teams = [t for t in teams if t is not None]

    # Colores por equipo (relleno). Si no coincide, usa gris.
    colors = {}
    if len(teams) >= 1:
        colors[str(teams[0])] = "#1f77b4"
    if len(teams) >= 2:
        colors[str(teams[1])] = "#ff7f0e"

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    # ejes consistentes
    ax.set_xlim(0, max(95, float(shots["t"].max() + 1)))

    for t in teams:
        team_color = colors.get(str(t), "#808080")
        s = shots[shots[team_col] == t].sort_values("t").copy()
        if s.empty:
            continue
        y = s["xg"].cumsum()
        ax.step(s["t"], y, where="post", linewidth=2.2, label=str(t))

        # marcar goles (si se detecta)
        goals = detect_goal_mask(s)
        if goals.any():
            sg = s[goals].copy()
            yg = y.loc[sg.index]
            ax.scatter(sg["t"], yg, marker="*", s=120, edgecolor="black", linewidth=0.7, zorder=5)

    # referencias
    ax.axvline(45, linestyle="--", linewidth=1, alpha=0.35)
    ax.axvline(90, linestyle="--", linewidth=1, alpha=0.35)

    # títulos y estilo
    ax.set_title(f"xG acumulado (por minuto) — {fmt_ctx(ctx)}", pad=12)
    ax.set_xlabel("Minuto")
    ax.set_ylabel("xG acumulado")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, loc="upper left")

    # anotaciones
    ymax = max(ax.get_ylim()[1], 0.01)
    ax.text(45, ymax*0.98, "HT", ha="right", va="top", fontsize=9, alpha=0.75)
    ax.text(90, ymax*0.98, "FT", ha="right", va="top", fontsize=9, alpha=0.75)
    ax.text(0.99, 0.02, "★ = gol", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.75)

    fig.tight_layout()
    if save and outdir is not None:
        fig.savefig(Path(outdir) / "02_xg_flow.png", bbox_inches="tight", dpi=180)
    return fig


def plot_passing_network(events_xy: pd.DataFrame, team: str, ctx: dict, save=False, outdir: Optional[Path] = None):
    """
    Passing network (Open Play, pases completados) en ventana 0–45' del 1T:
    - Nodo: posición media de origen del pase por jugador
    - Tamaño: volumen de pases (origen)
    - Aristas: pases completados entre jugador→receptor (filtra ruido)
    """
    team_col = detect_team_col(events_xy)
    player_col = detect_player_col(events_xy)

    passes = events_xy[events_xy["type"] == "Pass"].copy()
    if passes.empty:
        return
    passes = open_play_completed_passes(passes)
    passes = first_half_window(passes, max_min=45)

    # Columnas de receptor (dependen del endpoint / versión)
    rec_col = None
    for cand in ["pass_recipient", "pass_recipient_name"]:
        if cand in passes.columns:
            rec_col = cand
            break
    if rec_col is None:
        return

    p = passes[passes[team_col] == team].dropna(subset=["x", "y"]).copy()
    if p.empty:
        return
    p = normalize_attack_right(p, team=team, ctx=ctx)

    # Normalizar nombres si vienen como dicts
    def _name(v):
        if isinstance(v, dict):
            return v.get("name")
        return v

    p[player_col] = p[player_col].apply(_name)
    p[rec_col] = p[rec_col].apply(_name)

    p = p.dropna(subset=[player_col, rec_col]).copy()

    # Posición media de origen por jugador
    pos = (p.groupby(player_col)
             .agg(x_mean=("x", "mean"),
                  y_mean=("y", "mean"),
                  n=("x", "size"))
             .reset_index())

    # Conexiones jugador→receptor
    links = (p.groupby([player_col, rec_col])
               .size()
               .reset_index(name="count"))

    # Filtrar ruido: conexiones mínimas y jugadores con poco volumen
    links = links[links["count"] >= 2].copy()
    if links.empty:
        return
    keep_players = set(pos[pos["n"] >= 8][player_col])
    links = links[links[player_col].isin(keep_players) & links[rec_col].isin(keep_players)].copy()
    if links.empty:
        return

    # Merge posiciones de origen y receptor
    pos_map = pos.set_index(player_col)[["x_mean", "y_mean", "n"]].to_dict("index")
    links["sx"] = links[player_col].map(lambda k: pos_map.get(k, {}).get("x_mean"))
    links["sy"] = links[player_col].map(lambda k: pos_map.get(k, {}).get("y_mean"))
    links["ex"] = links[rec_col].map(lambda k: pos_map.get(k, {}).get("x_mean"))
    links["ey"] = links[rec_col].map(lambda k: pos_map.get(k, {}).get("y_mean"))
    links = links.dropna(subset=["sx", "sy", "ex", "ey"]).copy()

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11.5, 7.2))

    # Aristas: grosor ~ count (escalado)
    maxc = max(links["count"].max(), 1)
    widths = 0.6 + (links["count"] / maxc) * 4.0
    widths = widths.to_numpy()
    pitch.lines(links["sx"], links["sy"], links["ex"], links["ey"],
                lw=widths, ax=ax, alpha=0.35, zorder=2)

    # Nodos: tamaño ~ n
    maxn = max(pos["n"].max(), 1)
    node_sizes = 120 + (pos["n"] / maxn) * 900
    pitch.scatter(pos["x_mean"], pos["y_mean"],
                  s=node_sizes, ax=ax,
                  alpha=0.95, edgecolors="black", linewidth=0.4, zorder=3)

    # Etiquetas (apellido)
    for _, r in pos.iterrows():
        lab = str(r[player_col]).split()[-1]
        ax.text(r["x_mean"], r["y_mean"], lab, ha="center", va="center", fontsize=8, zorder=4)

    # Títulos / subtítulos
    ax.set_title(f"Passing network — {team} — {fmt_ctx(ctx)}", pad=14)
    ax.text(0.01, 0.01, "Filtros: Open Play + completados | Ventana: 1T 0–45' | Jugadores ≥8 pases | Enlaces ≥2",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.75)

    fig.tight_layout()
    if save and outdir is not None:
        fig.savefig(Path(outdir) / f"03_passing_network_{slug(team)}.png", bbox_inches="tight", dpi=180)
    return fig


def plot_progressive_passes(events_xy: pd.DataFrame, team: str, ctx: dict, save=False, outdir: Optional[Path] = None):
    """
    Progressive passes (Open Play, completados):
    - Progreso medido como reducción de distancia al centro de la portería rival.
    - Se muestran los TOP N por progreso para legibilidad.
    """
    team_col = detect_team_col(events_xy)

    passes = events_xy[events_xy["type"] == "Pass"].copy()
    if passes.empty:
        return
    passes = open_play_completed_passes(passes)
    passes = passes.dropna(subset=["x", "y", "pass_end_x", "pass_end_y"]).copy()

    p = passes[passes[team_col] == team].copy()
    if p.empty:
        return
    p = normalize_attack_right(p, team=team, ctx=ctx)

    # Distancia a centro de portería rival (x=120, y=40)
    p["start_d"] = np.sqrt((PITCH_L - p["x"])**2 + (40 - p["y"])**2)
    p["end_d"] = np.sqrt((PITCH_L - p["pass_end_x"])**2 + (40 - p["pass_end_y"])**2)
    p["progress"] = p["start_d"] - p["end_d"]

    # Umbral de progresión (metros). 10m suele ser buen baseline.
    prog = p[p["progress"] >= 10].copy()
    if prog.empty:
        return

    # Top por progreso (evita saturar)
    prog = prog.sort_values("progress", ascending=False).head(35).copy()

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11.5, 7.2))

    # Grosor ~ progreso (escalado)
    maxp = max(float(prog["progress"].max()), 1.0)
    widths = 0.8 + (prog["progress"] / maxp) * 2.6
    widths = widths.to_numpy()
    # Nota: matplotlib.quiver (usado internamente por mplsoccer) no soporta `width` vectorial.
    # Dibujamos flechas en un loop para permitir grosor variable por pase.
    for (sx, sy, ex, ey, w) in zip(
        prog["x"].to_numpy(), prog["y"].to_numpy(),
        prog["pass_end_x"].to_numpy(), prog["pass_end_y"].to_numpy(),
        widths
    ):
        pitch.arrows(
            float(sx), float(sy), float(ex), float(ey),
            ax=ax, width=float(w),
            headwidth=5, headlength=5,
            alpha=0.35, zorder=3
        )

    # Resumen
    ax.set_title(f"Progressive passes (Open Play, completados) — {team} — {fmt_ctx(ctx)}", pad=14)
    ax.text(0.01, 0.01,
            f"Mostrando: top {len(prog)} | Umbral: ≥10m | Progreso total: {prog['progress'].sum():.0f}m",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.75)

    fig.tight_layout()
    if save and outdir is not None:
        fig.savefig(Path(outdir) / f"04_progressive_passes_{slug(team)}.png", bbox_inches="tight", dpi=180)
    return fig


def plot_pressure_heatmap(events_xy: pd.DataFrame, team: str, ctx: dict, save=False, outdir: Optional[Path] = None):
    """
    Pressure heatmap (ubicación de presiones):
    - KDE suavizado para patrón espacial más interpretable (menos cuadriculado).
    - Incluye conteo total en subtítulo.
    """
    team_col = detect_team_col(events_xy)
    press = events_xy[events_xy["type"] == "Pressure"].dropna(subset=["x", "y"]).copy()
    press = press[press[team_col] == team].copy()
    if press.empty:
        return
    press = normalize_attack_right(press, team=team, ctx=ctx)

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11.5, 7.2))

    # KDE (si falla por alguna razón, fallback a bins)
    try:
        kde = pitch.kdeplot(press["x"].to_numpy(), press["y"].to_numpy(), ax=ax, fill=True, levels=60, alpha=0.9, zorder=1)
        cbar = plt.colorbar(kde, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Densidad de presiones")
    except Exception:
        bin_stat = pitch.bin_statistic(press["x"], press["y"], statistic="count", bins=(24, 16))
        hm = pitch.heatmap(bin_stat, ax=ax, alpha=0.85)
        cbar = plt.colorbar(hm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Conteo de presiones")

    ax.set_title(f"Presiones — mapa de calor — {team} — {fmt_ctx(ctx)}", pad=14)
    ax.text(0.01, 0.01, f"Total presiones: {len(press)} | Orientación: ambos equipos atacan →",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.75)

    fig.tight_layout()
    if save and outdir is not None:
        fig.savefig(Path(outdir) / f"05_pressure_heatmap_{slug(team)}.png", bbox_inches="tight", dpi=180)
    return fig



# ------------------------------------------------------------
# Match report (presentación): muestra todas las gráficas en una sola vista (grid 4x4)
# Nota: NO altera la lógica de cálculo de las gráficas; sólo compone una vista a partir de los PNG guardados.
# ------------------------------------------------------------
import numpy as np
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def _autocrop_white_arr(arr: np.ndarray, tol: int = 8, pad: int = 6) -> np.ndarray:
    """Recorta márgenes casi blancos de una imagen (numpy array)."""
    a = arr
    if a.ndim == 2:  # gray
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        rgb = a[..., :3]
        alpha = a[..., 3]
    else:
        rgb = a[..., :3]
        alpha = None

    # normaliza a uint8 0-255
    if rgb.dtype != np.uint8:
        rgb_u = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u = rgb

    white = (rgb_u[..., 0] >= 255 - tol) & (rgb_u[..., 1] >= 255 - tol) & (rgb_u[..., 2] >= 255 - tol)
    if alpha is not None:
        if alpha.dtype != np.uint8:
            a_u = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        else:
            a_u = alpha
        white = white | (a_u == 0)

    nonwhite = ~white
    if not np.any(nonwhite):
        return arr

    ys, xs = np.where(nonwhite)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]




# ---------- Streamlit helpers ----------
def generate_top5_streamlit(
    events_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    match_id: int,
    save_figs: bool,
    outdir: Path,
):
    ctx = match_context(matches_df, match_id)

    teams = [ctx.get("home_team"), ctx.get("away_team")]
    teams = [t for t in teams if t is not None]

    ev = prep_events_for_plots(events_df)

    figs = []

    fig1 = plot_shot_map(ev, ctx, save=save_figs, outdir=outdir)
    if fig1 is not None: figs.append(("Mapa de tiros", fig1))

    fig2 = plot_xg_flow(ev, ctx, save=save_figs, outdir=outdir)
    if fig2 is not None: figs.append(("xG acumulado", fig2))

    for t in teams:
        fig3 = plot_passing_network(ev, t, ctx, save=save_figs, outdir=outdir)
        if fig3 is not None: figs.append((f"Red de pases — {t}", fig3))

        fig4 = plot_progressive_passes(ev, t, ctx, save=save_figs, outdir=outdir)
        if fig4 is not None: figs.append((f"Pases progresivos — {t}", fig4))

        fig5 = plot_pressure_heatmap(ev, t, ctx, save=save_figs, outdir=outdir)
        if fig5 is not None: figs.append((f"Presiones — {t}", fig5))

    return ctx, teams, figs


def build_match_report_from_saved_pngs(outdir: Path, ctx: dict, teams: list[str]) -> dict:
    """Creates a single 'match report' PNG + PDF by composing the saved charts.
    Requires that save_figs=True so the PNGs exist in outdir.
    Returns dict with paths (png, pdf)."""
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec

    outdir = Path(outdir)

    def _slug(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or "team"

    home = teams[0] if len(teams) >= 1 else "Home"
    away = teams[1] if len(teams) >= 2 else "Away"

    files = [
        ("Mapa de tiros",               outdir / "01_shot_map.png"),
        ("xG acumulado",                outdir / "02_xg_flow.png"),
        (f"Red de pases — {home}",      outdir / f"03_passing_network_{_slug(home)}.png"),
        (f"Red de pases — {away}",      outdir / f"03_passing_network_{_slug(away)}.png"),
        (f"Pases progresivos — {home}", outdir / f"04_progressive_passes_{_slug(home)}.png"),
        (f"Pases progresivos — {away}", outdir / f"04_progressive_passes_{_slug(away)}.png"),
        (f"Presiones — {home}",         outdir / f"05_pressure_heatmap_{_slug(home)}.png"),
        (f"Presiones — {away}",         outdir / f"05_pressure_heatmap_{_slug(away)}.png"),
    ]

    # load helper
    def _read(path: Path):
        if not path.exists():
            return None
        try:
            return mpimg.imread(str(path))
        except Exception:
            return None

    imgs = [(title, _read(path), path.name) for title, path in files]

    fig = plt.figure(figsize=(18, 20), facecolor="white")
    gs = GridSpec(4, 4, figure=fig, left=0.03, right=0.97, top=0.94, bottom=0.03, wspace=0.06, hspace=0.10)

    positions = [
        (0, slice(0,2)), (0, slice(2,4)),
        (1, slice(0,2)), (1, slice(2,4)),
        (2, slice(0,2)), (2, slice(2,4)),
        (3, slice(0,2)), (3, slice(2,4)),
    ]

    missing = []
    for (panel_title, img, fname), (r, cspan) in zip(imgs, positions):
        ax = fig.add_subplot(gs[r, cspan])
        ax.axis("off")
        ax.set_title(panel_title, fontsize=12, pad=6, fontweight="semibold")
        if img is None:
            missing.append(fname)
            ax.text(0.5, 0.5, f"No se encontró:\n{fname}", ha="center", va="center", fontsize=11, color="#b00020")
        else:
            ax.imshow(img, aspect="auto")

    title = f"{home} vs {away}"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)

    subtitle_parts = []
    for k, lab in [("competition_id","Comp"), ("season_id","Season"), ("match_id","Match")]:
        if ctx.get(k) is not None:
            subtitle_parts.append(f"{lab} {ctx.get(k)}")
    subtitle = " | ".join(subtitle_parts)
    if subtitle:
        fig.text(0.5, 0.962, subtitle, ha="center", va="top", fontsize=11, color="#444444")

    if missing:
        fig.text(0.5, 0.015, "Aviso: faltan archivos en el reporte: " + ", ".join(missing),
                 ha="center", va="bottom", fontsize=9, color="#b00020")

    png_path = outdir / f"match_report_{_slug(home)}_vs_{_slug(away)}_match{ctx.get('match_id','na')}.png"
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path, "missing": missing}


def zip_folder(folder: Path) -> bytes:
    folder = Path(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))
    return buf.getvalue()


def main():
    st.title("StatsBomb Open Data — Match Report (Streamlit)")
    st.markdown(
        "Configura el partido y las salidas aquí abajo. "
        "Las opciones y botones están en el área principal (mobile-first)."
    )

    comps = get_competitions()

    # Detect typical columns
    cid_col = "competition_id" if "competition_id" in comps.columns else "competition"
    sid_col = "season_id" if "season_id" in comps.columns else "season"
    cn_col  = "competition_name" if "competition_name" in comps.columns else "competition"
    sn_col  = "season_name" if "season_name" in comps.columns else "season"

    comps_view = comps.copy()
    if cn_col in comps_view.columns and sn_col in comps_view.columns:
        comps_view = comps_view.sort_values([cn_col, sn_col])

    def comp_label(r):
        cn = str(r.get(cn_col, "")).strip()
        sn = str(r.get(sn_col, "")).strip()
        return f"{cn} | {sn} (cid={int(r[cid_col])}, sid={int(r[sid_col])})"

    comp_options = [(comp_label(r), (int(r[cid_col]), int(r[sid_col]))) for _, r in comps_view.iterrows()]

    st.markdown("### 1) Selecciona competición")
    with st.form("step1_competition", clear_on_submit=False):
        # Default selection (first item) so the user can just hit "Continuar"
        comp_choice = st.selectbox(
            "Competición / temporada",
            options=comp_options,
            format_func=lambda x: x[0],
            key="comp_choice",
        )
        go_next = st.form_submit_button("Continuar", use_container_width=True)

    # Persist selection only when user clicks "Continuar"
    if go_next or ("competition_id" not in st.session_state):
        st.session_state["competition_id"] = comp_choice[1][0]
        st.session_state["season_id"] = comp_choice[1][1]
        # Reset match selection when competition changes
        st.session_state.pop("match_id", None)

    competition_id = st.session_state.get("competition_id")
    season_id = st.session_state.get("season_id")

    matches = get_matches(competition_id, season_id)

    # Build match selector
    mid_col = "match_id" if "match_id" in matches.columns else None
    if mid_col is None:
        st.error("No se encontró match_id en el DataFrame de matches.")
        st.stop()

    def pick(cols, r):
        for c in cols:
            if c in matches.columns:
                return r.get(c)
        return None

    def match_label(r):
        ht = pick(["home_team_name","home_team"], r) or "Home"
        at = pick(["away_team_name","away_team"], r) or "Away"
        hs = pick(["home_score"], r)
        a_s = pick(["away_score"], r)
        date = pick(["match_date","date","kick_off"], r)
        score = f" ({hs}-{a_s})" if (hs is not None and a_s is not None and str(hs)!="nan" and str(a_s)!="nan") else ""
        meta = f" — {date}" if (date is not None and str(date)!="nan") else ""
        return f"{ht} vs {at}{score}{meta} | match_id={int(r[mid_col])}"

    matches_view = matches.copy()
    for dc in ["match_date","date","kick_off"]:
        if dc in matches_view.columns:
            matches_view = matches_view.sort_values(dc)
            break

    match_options = [(match_label(r), int(r[mid_col])) for _, r in matches_view.iterrows()]

    st.markdown("### 2) Selecciona partido y opciones")
    with st.form("step2_match_and_options", clear_on_submit=False):
        sel = st.selectbox(
            "Partido",
            options=match_options,
            format_func=lambda x: x[0],
            key="match_choice",
        )
        match_id = sel[1]

        st.markdown("**Opciones de salida**")
        o1, o2, o3, o4 = st.columns([1, 1, 1, 1], gap="medium")
        with o1:
            save_csv = st.checkbox("Guardar CSV", value=True, key="save_csv")
        with o2:
            save_parquet = st.checkbox("Guardar Parquet", value=True, key="save_parquet")
        with o3:
            save_figs = st.checkbox("Guardar gráficas PNG", value=True, key="save_figs")
        with o4:
            build_report = st.checkbox(
                "Crear Match Report (PNG+PDF)",
                value=True,
                help="Compone un dashboard final a partir de las PNG guardadas.",
                key="build_report",
            )

        run = st.form_submit_button("Generar", type="primary", use_container_width=True)

    # Only persist match_id when user submits the form (Generar)
    if run:
        st.session_state["match_id"] = match_id

    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.subheader("Vista rápida")

    with st.spinner("Descargando y procesando datos…"):
        events = get_events(match_id)
        lineups = get_lineups(match_id)

        ctx = match_context(matches, match_id)

        # Output folder (per-run)
        tmp_root = Path(tempfile.mkdtemp(prefix="statsbomb_app_"))
        outdir = tmp_root / f"{slug(ctx.get('home_team') or 'home')}_{slug(ctx.get('away_team') or 'away')}_{int(match_id)}"
        outdir.mkdir(parents=True, exist_ok=True)

        # Save data
        saved = {}
        saved.update({f"events_{k}": v for k,v in save_df(events, outdir, "events", save_csv=save_csv, save_parquet=save_parquet).items()})
        saved.update({f"lineups_{k}": v for k,v in save_df(lineups, outdir, "lineups", save_csv=save_csv, save_parquet=save_parquet).items()})
        # Save matches row too (handy)
        match_row = matches[matches[mid_col]==match_id].copy()
        saved.update({f"match_{k}": v for k,v in save_df(match_row, outdir, "match", save_csv=save_csv, save_parquet=save_parquet).items()})

        # Plots
        ctx, teams, figs = generate_top5_streamlit(events, matches, match_id, save_figs=save_figs, outdir=outdir)

        report_paths = None
        if build_report and save_figs:
            report_paths = build_match_report_from_saved_pngs(outdir, ctx={"competition_id": competition_id, "season_id": season_id, "match_id": match_id}, teams=teams)

    st.success("Listo ✅")

    st.markdown(f"### {fmt_ctx(ctx)}")

    # Display plots
    st.markdown("## Gráficas")
    for title, fig in figs:
        st.markdown(f"#### {title}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Show saved files + downloads
    st.markdown("## Archivos generados")
    files = sorted([p for p in outdir.rglob("*") if p.is_file()])
    st.write(f"Carpeta: `{outdir}`")
    st.write(f"Total archivos: **{len(files)}**")
    st.dataframe(pd.DataFrame({
        "archivo": [str(p.relative_to(outdir)) for p in files],
        "tamaño_kb": [round(p.stat().st_size/1024, 1) for p in files],
    }), use_container_width=True)

    zip_bytes = zip_folder(outdir)
    st.download_button(
        label="⬇️ Descargar TODO (ZIP)",
        data=zip_bytes,
        file_name=f"statsbomb_match_{match_id}.zip",
        mime="application/zip",
    )

    if report_paths:
        if report_paths.get("missing"):
            st.warning("Faltaron algunos PNG para el match report: " + ", ".join(report_paths["missing"]))
        for k in ["png","pdf"]:
            p = report_paths.get(k)
            if p and Path(p).exists():
                st.download_button(
                    label=f"⬇️ Descargar Match Report ({k.upper()})",
                    data=Path(p).read_bytes(),
                    file_name=Path(p).name,
                    mime="application/pdf" if k=="pdf" else "image/png",
                )

if __name__ == "__main__":
    main()
