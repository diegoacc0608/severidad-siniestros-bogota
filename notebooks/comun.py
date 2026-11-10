"""
Grupo 2 - Estadística Bayesiana - Entrega final
Módulo común a los notebooks 02 a 08: rutas, datos, celdas y métricas.
Es el mismo código de Grupo2_EntregaFinal_Bayesiana.py (secciones 1 y 6), separado
para que cada notebook pueda correr por sí solo.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import special as sp
from scipy import stats as st
import sys
import matplotlib

if "ipykernel" not in sys.modules:      # fuera de Jupyter/Colab no se abren ventanas
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- rutas
BASE = Path(__file__).resolve().parent                  # carpeta notebooks/
PROYECTO = BASE.parent.parent                           # .../Proyecto (solo en el computador del equipo)
SALIDAS = BASE / "salidas"
FIG = SALIDAS / "figuras"
RES = SALIDAS / "resultados"
DATOS = BASE / "datos"
for _c in (DATOS, FIG, RES):
    _c.mkdir(parents=True, exist_ok=True)


def ruta_datos(nombre):
    """Busca un archivo de datos en notebooks/datos/ y, si no está, en la carpeta del proyecto."""
    for d in (DATOS, PROYECTO / "0. Base de Datos" / "datos_trabajados", PROYECTO / "0. Base de Datos"):
        if (d / nombre).exists():
            return d / nombre
    raise FileNotFoundError(f"No encontré {nombre}: póngalo en {DATOS}")


SEED = 20261109
K, BURN, ITER = 4, 5000, 20000

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
AZUL, ROJO, GRIS, NARANJA = "#1f4e79", "#c0392b", "#7f7f7f", "#e67e22"
FRANJAS = ["Madrugada (0-6)", "Mañana (6-12)", "Tarde (12-18)", "Noche (18-24)"]
SUR = ["San Cristóbal", "Rafael Uribe Uribe", "Usme", "Ciudad Bolívar", "Bosa", "Tunjuelito"]

# ---------------------------------------------------------------- datos
df = pd.read_csv(ruta_datos("siniestros_limpios.csv"),
                 usecols=["anio", "hora", "franja", "franja3h", "localidad", "grave", "con_muertos"])
LOCS = sorted(df["localidad"].unique())
FR3 = sorted(df["franja3h"].unique())
L, F = len(LOCS), len(FRANJAS)


def celdas(d, col_franja="franja", niveles=FRANJAS, y="grave"):
    """Tabla completa localidad x franja (incluye celdas con n = 0)."""
    idx = pd.MultiIndex.from_product([LOCS, niveles], names=["localidad", col_franja])
    t = d.groupby(["localidad", col_franja])[y].agg(n="size", y="sum").reindex(idx, fill_value=0)
    return t.reset_index()


tr = df[df.anio.between(2015, 2017)]
te = df[df.anio.between(2018, 2019)]
d20 = df[df.anio == 2020]
c_tr, c_te, c_20 = celdas(tr), celdas(te), celdas(d20)
n_tr, y_tr = c_tr.n.values, c_tr.y.values
n_te, y_te = c_te.n.values, c_te.y.values
es_sur = np.isin(c_tr.localidad.values, SUR)
es_noche = np.isin(c_tr.franja.values, ["Madrugada (0-6)", "Noche (18-24)"])
eleg = (n_tr >= 30) & (n_te >= 30)          # celdas que cuentan para el orden (Avance 2)
idx_e = np.where(eleg)[0]
p_cruda_tr = np.where(n_tr > 0, y_tr / np.maximum(n_tr, 1), y_tr.sum() / n_tr.sum())
p_city = np.full(L * F, y_tr.sum() / n_tr.sum())
obs_te = np.where(n_te > 0, y_te / np.maximum(n_te, 1), np.nan)

# celdas por año (para M2)
cy = (tr.groupby(["localidad", "franja", "anio"]).grave.agg(n="size", y="sum")
      .reindex(pd.MultiIndex.from_product([LOCS, FRANJAS, [2015, 2016, 2017]]), fill_value=0))
N3 = cy["n"].values.reshape(L, F, 3)
Y3 = cy["y"].values.reshape(L, F, 3)


# ---------------------------------------------------------------- métricas
def metricas(p, n=n_te, y=y_te, eleg_=eleg, theta_thin=None):
    """Brier, log-loss, Spearman, top 10, MAE por subgrupo; con theta_thin (draws ya
    adelgazados 1 de cada 20) también lpd, cobertura del 90 % y piso binomial."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    N = n.sum()
    brier = float((y * (1 - p) ** 2 + (n - y) * p ** 2).sum() / N)
    logloss = float(-(y * np.log(p) + (n - y) * np.log1p(-p)).sum() / N)
    obs = np.where(n > 0, y / np.maximum(n, 1), np.nan)
    rho = float(st.spearmanr(p[eleg_], obs[eleg_])[0])
    idx = np.where(eleg_)[0]
    top_p = set(idx[np.argsort(-p[idx])[:10]])
    top_o = set(idx[np.argsort(-obs[idx])[:10]])
    r = dict(brier=brier, logloss=logloss, spearman=rho, top10=len(top_p & top_o))
    err = p - obs
    if len(p) != L * F:
        return r
    for nombre, g1 in (("sur", es_sur), ("noche_madrug", es_noche), ("pequenas", n_tr < 500)):
        m1_, m0_ = eleg_ & g1, eleg_ & ~g1
        r[f"mae_{nombre}"] = float(np.abs(err[m1_]).mean())
        r[f"mae_no_{nombre}"] = float(np.abs(err[m0_]).mean())
        r[f"sesgo_{nombre}"] = float(err[m1_].mean())
        r[f"sesgo_no_{nombre}"] = float(err[m0_].mean())
    r["mae_total"] = float(np.abs(err[eleg_]).mean())
    r["sesgo_total"] = float(err[eleg_].mean())
    if theta_thin is not None:
        th = np.clip(theta_thin.astype(float), 1e-9, 1 - 1e-9)
        lpd = st.binom.logpmf(y[None, :], n[None, :], th)
        r["lpd"] = float((sp.logsumexp(lpd, axis=0) - np.log(th.shape[0])).sum())
        rng = np.random.default_rng(1)
        ysim = rng.binomial(n[None, :], th)
        lo_, hi_ = np.quantile(ysim, [0.05, 0.95], axis=0)
        dentro = (y >= lo_) & (y <= hi_)
        r["cobertura90"] = float(dentro[n > 0].mean())
        r["cobertura90_pequenas"] = float(dentro[eleg_ & (n_tr < 500)].mean())
        r["cobertura90_grandes"] = float(dentro[eleg_ & (n_tr >= 500)].mean())
        pp = np.clip(p - r["sesgo_total"], 1e-3, 1 - 1e-3)
        piso = np.sqrt(2 / np.pi) * np.sqrt(pp * (1 - pp) / np.maximum(n, 1))
        r["piso_mae_pequenas"] = float(piso[eleg_ & (n_tr < 500)].mean())
        r["piso_mae_grandes"] = float(piso[eleg_ & (n_tr >= 500)].mean())
    return r


def prob_top10(th):
    """P(celda en el top 10) entre las celdas elegibles; usa 1 de cada 10 draws."""
    sub = th[::10][:, idx_e]
    rk = (-sub).argsort(1).argsort(1)
    pt = np.zeros(L * F)
    pt[idx_e] = (rk < 10).mean(0)
    return pt


def stats_celdas(yc, nc):
    m_ = nc >= 30
    pr = yc[..., m_] / nc[m_]
    return np.stack([pr.mean(-1), pr.std(-1), pr.max(-1) - pr.min(-1), (pr > 0.5).mean(-1)], -1)


def limpio(o):
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpio(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def guardar_json(R, nombre):
    with open(RES / nombre, "w", encoding="utf-8") as fh:
        json.dump(limpio(R), fh, ensure_ascii=False, indent=1)
    print("Guardado:", RES / nombre)
