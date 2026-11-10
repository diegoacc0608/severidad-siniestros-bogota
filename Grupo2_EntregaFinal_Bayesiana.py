"""
Grupo 2 - Estadística Bayesiana - Entrega final
Severidad relativa de los siniestros viales en Bogotá por localidad y franja horaria.

Datos: Secretaría Distrital de Movilidad, "Siniestros viales consolidados Bogotá D.C."
copia descargada por el equipo del portal Datos Abiertos Bogotá (al 9-nov-2026 el recurso
aparece restringido). Versión vigente de la misma fuente (SDM, "Siniestralidad BD"):
https://datos.movilidadbogota.gov.co/datasets/ea243e7de8e846c8bd27e47c08771d66
(se usa la versión limpia `siniestros_limpios.csv`, construida en el notebook de EDA).

Modelos (MCMC programado a mano; ArviZ solo para R-hat y ESS):
  M1  Beta-Binomial jerárquico. Gibbs exacto para theta_c y Metropolis-Hastings
      (paseo aleatorio en logit(m), log(kappa)) para los hiperparámetros.
  M2  Regresión logística jerárquica: logit theta_{l,f,t} = mu + a_l + b_f + g_{lf} + d_t.
      Metropolis-Hastings por componentes (vectorizado), movimientos de traslación
      entre niveles y Gibbs (Inversa-Gamma conjugada) para las varianzas.

Partición: se estima con 2015-2017 y se valida una sola vez con 2018-2019. 2020 aparte.

Ejecutar desde cualquier carpeta:
  uv run --no-project --with numpy --with pandas --with scipy --with matplotlib \
         --with "arviz<1" python Grupo2_EntregaFinal_Bayesiana.py
Salidas: figuras/ (PNG) y resultados/ (CSV y JSON con todas las cifras del informe).
Versiones con las que se generó el informe: Python 3.12.10, numpy 2.3.5, pandas 2.3.3,
scipy 1.18.1, matplotlib 3.10.7, arviz 0.23.4. Copia local de la base: 25 de octubre de 2026.
Si se entrega en Teams, poner `siniestros_limpios.csv` en la misma carpeta que este script.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy import special as sp
from scipy import stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import arviz as az  # noqa: E402

BASE = Path(__file__).resolve().parent
# Busca la base limpia junto al script (entrega en Teams) o en la carpeta del proyecto
DATOS = next((d for d in (BASE, BASE / "datos", BASE.parent / "0. Base de Datos" / "datos_trabajados")
              if (d / "siniestros_limpios.csv").exists()), BASE)
FIG = BASE / "figuras"
RES = BASE / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

SEED = 20261109
K = 4            # cadenas
BURN = 5000      # iteraciones de calentamiento (adaptación de la propuesta)
ITER = 20000     # iteraciones guardadas por cadena

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
AZUL, ROJO, GRIS, NARANJA = "#1f4e79", "#c0392b", "#7f7f7f", "#e67e22"

FRANJAS = ["Madrugada (0-6)", "Mañana (6-12)", "Tarde (12-18)", "Noche (18-24)"]
SUR = ["San Cristóbal", "Rafael Uribe Uribe", "Usme", "Ciudad Bolívar", "Bosa", "Tunjuelito"]
R = {}  # todas las cifras que se reportan

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
df = pd.read_csv(DATOS / "siniestros_limpios.csv",
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

# control: coincide con las celdas ya publicadas en datos_trabajados
if (DATOS / "celdas_ajuste_2015_2017.csv").exists():
    ref = pd.read_csv(DATOS / "celdas_ajuste_2015_2017.csv")
    chk = c_tr.merge(ref, on=["localidad", "franja"])
    assert (chk.n_x == chk.n_y).all() and (chk.y == chk.graves).all(), "celdas no coinciden"

n_tr, y_tr = c_tr.n.values, c_tr.y.values
n_te, y_te = c_te.n.values, c_te.y.values
loc_idx = np.repeat(np.arange(L), F)
fr_idx = np.tile(np.arange(F), L)
es_sur = np.isin(c_tr.localidad.values, SUR)
es_noche = np.isin(c_tr.franja.values, ["Madrugada (0-6)", "Noche (18-24)"])
eleg = (n_tr >= 30) & (n_te >= 30)  # celdas que cuentan para el orden (decidido en el Avance 2)

R["datos"] = dict(n_total=int(len(df)), n_tr=int(n_tr.sum()), y_tr=int(y_tr.sum()),
                  n_te=int(n_te.sum()), y_te=int(y_te.sum()), n_20=int(len(d20)),
                  y_20=int(d20.grave.sum()), p_tr=float(y_tr.sum() / n_tr.sum()),
                  p_te=float(y_te.sum() / n_te.sum()), p_20=float(d20.grave.mean()),
                  celdas=int(L * F), celdas_eleg=int(eleg.sum()),
                  celdas_n0=int((n_tr == 0).sum()), celdas_peq=int((n_tr < 500).sum()))

# ---------------------------------------------------------------------------
# 2. Figuras exploratorias (EDA)
# ---------------------------------------------------------------------------
base = df[df.anio <= 2019]
h = base.groupby("hora").grave.agg(["size", "mean"])
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.bar(h.index, h["size"], color="#9dc3e6")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Número de siniestros")
for x in (6, 12, 18):
    ax.axvline(x - 0.5, ls=":", color=GRIS)
ax2 = ax.twinx()
ax2.plot(h.index, 100 * h["mean"], color=ROJO, marker="o", ms=3)
ax2.set_ylabel("% graves", color=ROJO)
ax.set_title("Siniestros y % de graves por hora, 2015-2019")
fig.tight_layout()
fig.savefig(FIG / "eda_hora.png")
plt.close(fig)

lo = base.groupby("localidad").grave.agg(["size", "mean"]).drop("Sumapaz")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.scatter(lo["size"], 100 * lo["mean"], c=[ROJO if i in SUR else AZUL for i in lo.index], s=22)
for nom, r in lo.iterrows():
    ax.annotate(nom, (r["size"], 100 * r["mean"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
ax.axhline(100 * base.grave.mean(), ls="--", color=GRIS)
ax.set_xlabel("Número de siniestros 2015-2019")
ax.set_ylabel("% graves")
rho_loc = st.spearmanr(lo["size"], lo["mean"])[0]
ax.set_title(f"Más siniestros no significa más graves (Spearman = {rho_loc:.2f}; rojo: sur)")
fig.tight_layout()
fig.savefig(FIG / "eda_localidad.png")
plt.close(fig)
R["eda_rho_localidad"] = float(rho_loc)

mat = (c_tr.y / c_tr.n.replace(0, np.nan)).values.reshape(L, F)
orden_loc = np.argsort(-np.nanmean(mat, axis=1))
fig, ax = plt.subplots(figsize=(4.6, 4.6))
im = ax.imshow(100 * mat[orden_loc], cmap="Reds", aspect="auto", vmin=15, vmax=60)
ax.set_xticks(range(F), ["Madrug.", "Mañana", "Tarde", "Noche"])
ax.set_yticks(range(L), [LOCS[i] for i in orden_loc], fontsize=7)
for i, li in enumerate(orden_loc):
    for j in range(F):
        v = mat[li, j]
        ax.text(j, i, "-" if np.isnan(v) else f"{100 * v:.0f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, label="% graves", shrink=0.7)
ax.set_title("Proporción cruda de graves por celda, 2015-2017")
fig.tight_layout()
fig.savefig(FIG / "eda_celdas.png")
plt.close(fig)

an = df.groupby("anio").grave.agg(["size", "mean"])
fig, ax = plt.subplots(figsize=(6.4, 2.4))
cols = [AZUL] * 3 + [NARANJA] * 2 + [GRIS]
ax.bar(an.index, 100 * an["mean"], color=cols)
for x, v, n in zip(an.index, an["mean"], an["size"]):
    ax.text(x, 100 * v + 0.5, f"{100 * v:.1f} %\n(n={n:,})".replace(",", "."), ha="center", fontsize=7)
ax.set_ylim(0, 45)
ax.set_ylabel("% graves")
ax.set_title("Proporción de graves por año (azul: ajuste; naranja: prueba; gris: 2020)")
fig.tight_layout()
fig.savefig(FIG / "eda_anio.png")
plt.close(fig)
R["eda_anio"] = {int(k): float(v) for k, v in an["mean"].items()}


# ---------------------------------------------------------------------------
# 3. Utilidades
# ---------------------------------------------------------------------------
def adaptar(logs, acc, it):
    """Ajusta la escala de la propuesta cada 100 iteraciones del calentamiento (objetivo ~0,3-0,44)."""
    if it < BURN and it > 0 and it % 100 == 0:
        tasa = acc / 100.0
        logs += np.where(tasa > 0.40, 0.15, np.where(tasa < 0.25, -0.15, 0.0))
        acc[...] = 0
    return logs


def resumen_az(post):
    idata = az.from_dict(posterior=post)
    s = az.summary(idata, kind="diagnostics")
    return s


# ---------------------------------------------------------------------------
# 4. M1: Beta-Binomial jerárquico (Gibbs + Metropolis-Hastings)
#   y_c | theta_c ~ Bin(n_c, theta_c)
#   theta_c | m, k ~ Beta(m k, (1-m) k)
#   m ~ Beta(2, 4),  k ~ Gamma(2, tasa)   (tasa 0,04 -> media 50)
# ---------------------------------------------------------------------------
def m1_logpost_hiper(u, v, theta, a_m=2.0, b_m=4.0, sh_k=2.0, rt_k=0.04):
    m, k = sp.expit(u), np.exp(v)
    al, be = m * k, (1 - m) * k
    lt, l1t = np.log(theta), np.log1p(-theta)
    ll = ((al - 1)[:, None] * lt + (be - 1)[:, None] * l1t).sum(1) - theta.shape[1] * sp.betaln(al, be)
    lp = (a_m - 1) * np.log(m) + (b_m - 1) * np.log1p(-m) + (sh_k - 1) * v - rt_k * k
    jac = np.log(m) + np.log1p(-m) + v
    return ll + lp + jac


def correr_m1(n, y, rt_k=0.04, a_m=2.0, b_m=4.0, seed=SEED, guardar_theta=True):
    rng = np.random.default_rng(seed)
    C = len(n)
    u = rng.normal(sp.logit(0.33), 0.3, K)
    v = np.log(rng.uniform(20, 100, K))
    logs = np.log(np.array([0.05, 0.15]))
    acc = np.zeros(2)
    out = {"m": np.empty((K, ITER)), "kappa": np.empty((K, ITER))}
    if guardar_theta:
        out["theta"] = np.empty((K, ITER, C), dtype=np.float32)
    nacc = 0
    for it in range(BURN + ITER):
        m, k = sp.expit(u), np.exp(v)
        # Paso de Gibbs: theta_c | m, k, y ~ Beta(m k + y_c, (1-m) k + n_c - y_c)
        theta = rng.beta((m * k)[:, None] + y, ((1 - m) * k)[:, None] + n - y)
        theta = np.clip(theta, 1e-12, 1 - 1e-12)
        # Paso de MH: (logit m, log k) con paseo aleatorio normal
        up = u + np.exp(logs[0]) * rng.standard_normal(K)
        vp = v + np.exp(logs[1]) * rng.standard_normal(K)
        lr = (m1_logpost_hiper(up, vp, theta, a_m, b_m, 2.0, rt_k)
              - m1_logpost_hiper(u, v, theta, a_m, b_m, 2.0, rt_k))
        ok = np.log(rng.uniform(size=K)) < lr
        u, v = np.where(ok, up, u), np.where(ok, vp, v)
        acc += ok.mean()
        logs = adaptar(logs, acc, it)
        if it >= BURN:
            j = it - BURN
            nacc += ok.sum()
            out["m"][:, j], out["kappa"][:, j] = sp.expit(u), np.exp(v)
            if guardar_theta:
                out["theta"][:, j] = theta
    out["acept"] = nacc / (K * ITER)
    return out


# ---------------------------------------------------------------------------
# 5. M2: logística jerárquica
#   y_{lft} | theta ~ Bin(n_{lft}, theta_{lft}),  logit theta = mu + a_l + b_f + g_lf + d_t
#   mu ~ N(logit(1/3), 1);  a_l ~ N(0, s2a);  b_f ~ N(0, s2b);  g_lf ~ N(0, s2g);  d_t ~ N(0, s2d)
#   s2a ~ IG(3, 0.5), s2b ~ IG(3, 0.5), s2g ~ IG(3, 0.1), s2d ~ IG(3, 0.05)
# ---------------------------------------------------------------------------
M0, S0 = sp.logit(1 / 3), 1.0
IG = {"a": (3.0, 0.5), "b": (3.0, 0.5), "g": (3.0, 0.1), "d": (3.0, 0.05)}


def correr_m2(Y, N, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    T = Y.shape[2]
    mu = rng.normal(M0, 0.2, K)
    a = rng.normal(0, 0.2, (K, L))
    b = rng.normal(0, 0.2, (K, F))
    g = rng.normal(0, 0.1, (K, L, F))
    d = rng.normal(0, 0.05, (K, T))
    s2 = {key: np.full(K, IG[key][1] / (IG[key][0] - 1)) for key in IG}

    def eta_de(mu, a, b, g, d):
        return mu[:, None, None, None] + a[:, :, None, None] + b[:, None, :, None] + g[..., None] + d[:, None, None, :]

    def ll_de(eta):
        return Y * eta - N * np.logaddexp(0.0, eta)

    escalas = {"mu": np.full(1, np.log(0.02)), "a": np.full(L, np.log(0.05)), "b": np.full(F, np.log(0.02)),
               "g": np.full((L, F), np.log(0.05)), "d": np.full(T, np.log(0.02)),
               "sh_mu_a": np.full(1, np.log(0.1)), "sh_mu_b": np.full(1, np.log(0.1)),
               "sh_mu_d": np.full(1, np.log(0.1)), "sh_a_g": np.full(L, np.log(0.1)),
               "sh_b_g": np.full(F, np.log(0.1))}
    acc = {k_: np.zeros_like(v_) for k_, v_ in escalas.items()}
    tasas = {k_: np.zeros_like(v_) for k_, v_ in escalas.items()}
    out = {k_: [] for k_ in ["mu", "sigma_a", "sigma_b", "sigma_g", "sigma_d"]}
    out["theta_new"] = np.empty((K, ITER, L * F), dtype=np.float32)
    out["a"] = np.empty((K, ITER, L), dtype=np.float32)
    out["b"] = np.empty((K, ITER, F), dtype=np.float32)
    out["d"] = np.empty((K, ITER, T), dtype=np.float32)

    eta = eta_de(mu, a, b, g, d)
    ll = ll_de(eta)
    for it in range(BURN + ITER):
        # --- mu
        e = np.exp(escalas["mu"][0]) * rng.standard_normal(K)
        eta_p = eta + e[:, None, None, None]
        llp = ll_de(eta_p)
        lr = (llp - ll).sum((1, 2, 3)) - ((mu + e - M0) ** 2 - (mu - M0) ** 2) / (2 * S0 ** 2)
        ok = np.log(rng.uniform(size=K)) < lr
        mu = np.where(ok, mu + e, mu)
        eta = np.where(ok[:, None, None, None], eta_p, eta)
        ll = np.where(ok[:, None, None, None], llp, ll)
        acc["mu"] += ok.mean()
        tasas["mu"] += ok.mean()
        # --- a_l (20 componentes; cada una solo afecta a sus filas)
        e = np.exp(escalas["a"]) * rng.standard_normal((K, L))
        eta_p = eta + e[:, :, None, None]
        llp = ll_de(eta_p)
        lr = (llp - ll).sum((2, 3)) - ((a + e) ** 2 - a ** 2) / (2 * s2["a"][:, None])
        ok = np.log(rng.uniform(size=(K, L))) < lr
        a = np.where(ok, a + e, a)
        m4 = ok[:, :, None, None]
        eta, ll = np.where(m4, eta_p, eta), np.where(m4, llp, ll)
        acc["a"] += ok.mean(0)
        tasas["a"] += ok.mean(0)
        # --- b_f
        e = np.exp(escalas["b"]) * rng.standard_normal((K, F))
        eta_p = eta + e[:, None, :, None]
        llp = ll_de(eta_p)
        lr = (llp - ll).sum((1, 3)) - ((b + e) ** 2 - b ** 2) / (2 * s2["b"][:, None])
        ok = np.log(rng.uniform(size=(K, F))) < lr
        b = np.where(ok, b + e, b)
        m4 = ok[:, None, :, None]
        eta, ll = np.where(m4, eta_p, eta), np.where(m4, llp, ll)
        acc["b"] += ok.mean(0)
        tasas["b"] += ok.mean(0)
        # --- g_lf (80 componentes)
        e = np.exp(escalas["g"]) * rng.standard_normal((K, L, F))
        eta_p = eta + e[..., None]
        llp = ll_de(eta_p)
        lr = (llp - ll).sum(3) - ((g + e) ** 2 - g ** 2) / (2 * s2["g"][:, None, None])
        ok = np.log(rng.uniform(size=(K, L, F))) < lr
        g = np.where(ok, g + e, g)
        m4 = ok[..., None]
        eta, ll = np.where(m4, eta_p, eta), np.where(m4, llp, ll)
        acc["g"] += ok.mean(0)
        tasas["g"] += ok.mean(0)
        # --- d_t
        e = np.exp(escalas["d"]) * rng.standard_normal((K, T))
        eta_p = eta + e[:, None, None, :]
        llp = ll_de(eta_p)
        lr = (llp - ll).sum((1, 2)) - ((d + e) ** 2 - d ** 2) / (2 * s2["d"][:, None])
        ok = np.log(rng.uniform(size=(K, T))) < lr
        d = np.where(ok, d + e, d)
        m4 = ok[:, None, None, :]
        eta, ll = np.where(m4, eta_p, eta), np.where(m4, llp, ll)
        acc["d"] += ok.mean(0)
        tasas["d"] += ok.mean(0)
        # --- Traslaciones (la verosimilitud no cambia; solo la razón de priors)
        #     mu + c, efectos - c  : reduce la correlación entre el intercepto y los efectos
        for clave, vec, s2k in (("sh_mu_a", a, "a"), ("sh_mu_b", b, "b"), ("sh_mu_d", d, "d")):
            c = np.exp(escalas[clave][0]) * rng.standard_normal(K)
            mu_p, vec_p = mu + c, vec - c[:, None]
            lr = (-((mu_p - M0) ** 2 - (mu - M0) ** 2) / (2 * S0 ** 2)
                  - ((vec_p ** 2).sum(1) - (vec ** 2).sum(1)) / (2 * s2[s2k]))
            ok = np.log(rng.uniform(size=K)) < lr
            mu = np.where(ok, mu_p, mu)
            vec[...] = np.where(ok[:, None], vec_p, vec)
            acc[clave] += ok.mean()
            tasas[clave] += ok.mean()
        #     a_l + c_l, g_l. - c_l   y   b_f + c_f, g_.f - c_f
        c = np.exp(escalas["sh_a_g"]) * rng.standard_normal((K, L))
        a_p, g_p = a + c, g - c[:, :, None]
        lr = (-((a_p ** 2 - a ** 2) / (2 * s2["a"][:, None]))
              - ((g_p ** 2 - g ** 2).sum(2) / (2 * s2["g"][:, None])))
        ok = np.log(rng.uniform(size=(K, L))) < lr
        a, g = np.where(ok, a_p, a), np.where(ok[:, :, None], g_p, g)
        acc["sh_a_g"] += ok.mean(0)
        tasas["sh_a_g"] += ok.mean(0)
        c = np.exp(escalas["sh_b_g"]) * rng.standard_normal((K, F))
        b_p, g_p = b + c, g - c[:, None, :]
        lr = (-((b_p ** 2 - b ** 2) / (2 * s2["b"][:, None]))
              - ((g_p ** 2 - g ** 2).sum(1) / (2 * s2["g"][:, None])))
        ok = np.log(rng.uniform(size=(K, F))) < lr
        b, g = np.where(ok, b_p, b), np.where(ok[:, None, :], g_p, g)
        acc["sh_b_g"] += ok.mean(0)
        tasas["sh_b_g"] += ok.mean(0)
        # --- Gibbs para las varianzas: s2 | x ~ IG(a0 + J/2, b0 + sum(x^2)/2)
        for key, x in (("a", a), ("b", b), ("g", g.reshape(K, -1)), ("d", d)):
            a0, b0 = IG[key]
            s2[key] = 1.0 / rng.gamma(a0 + x.shape[1] / 2, 1.0 / (b0 + (x ** 2).sum(1) / 2))
        # --- adaptación de escalas durante el calentamiento
        if 0 < it < BURN and it % 100 == 0:
            for k_ in escalas:
                tasa = acc[k_] / 100.0
                escalas[k_] = escalas[k_] + np.where(tasa > 0.44, 0.15, np.where(tasa < 0.25, -0.15, 0.0))
                acc[k_][...] = 0
        if it == BURN - 1:
            for k_ in tasas:
                tasas[k_][...] = 0
        if it >= BURN:
            j = it - BURN
            out["mu"].append(mu.copy())
            for key in "abgd":
                out["sigma_" + key].append(np.sqrt(s2[key]))
            # severidad esperada en un año nuevo: d_new ~ N(0, s2d)
            d_new = np.sqrt(s2["d"]) * rng.standard_normal(K)
            eta_new = mu[:, None, None] + a[:, :, None] + b[:, None, :] + g + d_new[:, None, None]
            out["theta_new"][:, j] = sp.expit(eta_new).reshape(K, -1)
            out["a"][:, j], out["b"][:, j], out["d"][:, j] = a, b, d
    for k_ in ["mu", "sigma_a", "sigma_b", "sigma_g", "sigma_d"]:
        out[k_] = np.stack(out[k_], axis=1)
    out["acept"] = {k_: float(np.mean(v_) / ITER) for k_, v_ in tasas.items()}
    return out


# ---------------------------------------------------------------------------
# 6. Métricas de validación
# ---------------------------------------------------------------------------
def metricas(p, n=n_te, y=y_te, eleg_=eleg, theta_draws=None):
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
    if len(p) != L * F:  # sensibilidad con 160 celdas: sin subgrupos
        return r
    for nombre, g1 in (("sur", es_sur), ("noche_madrug", es_noche), ("pequenas", n_tr < 500)):
        m1_, m0_ = eleg_ & g1, eleg_ & ~g1
        r[f"mae_{nombre}"] = float(np.abs(err[m1_]).mean())
        r[f"mae_no_{nombre}"] = float(np.abs(err[m0_]).mean())
        r[f"sesgo_{nombre}"] = float(err[m1_].mean())
        r[f"sesgo_no_{nombre}"] = float(err[m0_].mean())
    r["mae_total"] = float(np.abs(err[eleg_]).mean())
    r["sesgo_total"] = float(err[eleg_].mean())
    if theta_draws is not None:
        # densidad predictiva logarítmica y cobertura del intervalo predictivo del 90 %
        th = theta_draws.reshape(-1, theta_draws.shape[-1])[::20].astype(float)
        th = np.clip(th, 1e-9, 1 - 1e-9)
        lpd = st.binom.logpmf(y[None, :], n[None, :], th)
        r["lpd"] = float((sp.logsumexp(lpd, axis=0) - np.log(th.shape[0])).sum())
        rng = np.random.default_rng(1)
        ysim = rng.binomial(n[None, :], th)
        lo_, hi_ = np.quantile(ysim, [0.05, 0.95], axis=0)
        dentro = (y >= lo_) & (y <= hi_)
        r["cobertura90"] = float(dentro[n > 0].mean())
        r["cobertura90_pequenas"] = float(dentro[eleg_ & (n_tr < 500)].mean())
        r["cobertura90_grandes"] = float(dentro[eleg_ & (n_tr >= 500)].mean())
        # error esperado solo por el azar binomial de 2018-2019 (piso de ruido del MAE)
        pp = np.clip(p - r["sesgo_total"], 1e-3, 1 - 1e-3)
        piso = np.sqrt(2 / np.pi) * np.sqrt(pp * (1 - pp) / np.maximum(n, 1))
        r["piso_mae_pequenas"] = float(piso[eleg_ & (n_tr < 500)].mean())
        r["piso_mae_grandes"] = float(piso[eleg_ & (n_tr >= 500)].mean())
    return r


# ---------------------------------------------------------------------------
# 7. Chequeo predictivo a priori (M1 y M2)
# ---------------------------------------------------------------------------
def stats_celdas(yc, nc):
    m_ = nc >= 30
    pr = yc[..., m_] / nc[m_]
    return np.stack([pr.mean(-1), pr.std(-1), pr.max(-1) - pr.min(-1), (pr > 0.5).mean(-1)], -1)


rng = np.random.default_rng(SEED + 7)
S = 4000
mpri = rng.beta(2, 4, S)
kpri = rng.gamma(2, 1 / 0.04, S)
tpri = rng.beta((mpri * kpri)[:, None], ((1 - mpri) * kpri)[:, None], (S, L * F))
st_m1 = stats_celdas(rng.binomial(n_tr, tpri), n_tr)

cy = (tr.groupby(["localidad", "franja", "anio"]).grave.agg(n="size", y="sum")
      .reindex(pd.MultiIndex.from_product([LOCS, FRANJAS, [2015, 2016, 2017]]), fill_value=0))
N3 = cy["n"].values.reshape(L, F, 3)
Y3 = cy["y"].values.reshape(L, F, 3)


def ig(a0, b0, size):
    return 1 / rng.gamma(a0, 1 / b0, size)


mu_p = rng.normal(M0, S0, S)
eta_pri = (mu_p[:, None, None, None]
           + np.sqrt(ig(*IG["a"], S))[:, None, None, None] * rng.standard_normal((S, L, 1, 1))
           + np.sqrt(ig(*IG["b"], S))[:, None, None, None] * rng.standard_normal((S, 1, F, 1))
           + np.sqrt(ig(*IG["g"], S))[:, None, None, None] * rng.standard_normal((S, L, F, 1))
           + np.sqrt(ig(*IG["d"], S))[:, None, None, None] * rng.standard_normal((S, 1, 1, 3)))
y2 = rng.binomial(N3[None], sp.expit(eta_pri)).sum(-1).reshape(S, -1)
st_m2 = stats_celdas(y2, n_tr)
st_obs = stats_celdas(y_tr[None], n_tr)[0]
nombres = ["Media entre celdas", "Desviación entre celdas", "Rango (máx - mín)", "Fracción de celdas > 50 %"]
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4))
ppc = {}
for fila, (stt, nom) in enumerate(((st_m1, "M1"), (st_m2, "M2"))):
    for j in range(4):
        ax = axs[fila, j]
        ax.hist(stt[:, j], bins=40, color="#9dc3e6")
        ax.axvline(st_obs[j], color=ROJO)
        ax.set_yticks([])
        if fila == 0:
            ax.set_title(nombres[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(nom)
        ppc[f"{nom}_{j}"] = float((stt[:, j] < st_obs[j]).mean())
fig.suptitle("Chequeo predictivo a priori (histograma: 4.000 ciudades simuladas; rojo: 2015-2017)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "prior_predictivo.png")
plt.close(fig)
R["ppc_percentil"] = ppc
R["ppc_obs"] = st_obs.tolist()

# ---------------------------------------------------------------------------
# 8. Ajuste de M1 y M2
# ---------------------------------------------------------------------------
t0 = time.time()
m1 = correr_m1(n_tr, y_tr)
t1 = time.time()
m2 = correr_m2(Y3, N3)
t2 = time.time()
R["tiempo_s"] = dict(M1=round(t1 - t0, 1), M2=round(t2 - t1, 1))
R["acept_M1"] = float(m1["acept"])
R["acept_M2"] = m2["acept"]

# Diagnósticos con ArviZ (solo R-hat y ESS)
i_cel = {f"{c_tr.localidad[i]} | {c_tr.franja[i]}": i for i in range(L * F)}
mira = ["Kennedy | Noche (18-24)", "La Candelaria | Madrugada (0-6)", "Usme | Noche (18-24)"]
diag1 = resumen_az({"m": m1["m"], "kappa": m1["kappa"], "theta": m1["theta"].astype(float)})
diag2 = resumen_az({"mu": m2["mu"], "sigma_a": m2["sigma_a"], "sigma_b": m2["sigma_b"],
                    "sigma_g": m2["sigma_g"], "sigma_d": m2["sigma_d"],
                    "theta_new": m2["theta_new"].astype(float)})
diag1.to_csv(RES / "diagnosticos_M1.csv")
diag2.to_csv(RES / "diagnosticos_M2.csv")
R["diag"] = {}
for nom, dg in (("M1", diag1), ("M2", diag2)):
    R["diag"][nom] = dict(rhat_max=float(dg["r_hat"].max()), ess_bulk_min=float(dg["ess_bulk"].min()),
                          ess_tail_min=float(dg["ess_tail"].min()),
                          hiper={i: dict(rhat=float(dg.loc[i, "r_hat"]), ess=float(dg.loc[i, "ess_bulk"]))
                                 for i in dg.index if "[" not in i})

# Trazas
fig, axs = plt.subplots(3, 2, figsize=(7.2, 4.8))
series = [("M1: m", m1["m"]), ("M1: κ", m1["kappa"]),
          ("M1: θ Usme | Noche", m1["theta"][:, :, i_cel[mira[2]]]),
          ("M2: μ", m2["mu"]), ("M2: σ_γ (interacción)", m2["sigma_g"]), ("M2: σ_δ (año)", m2["sigma_d"])]
colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for ax, (tit, x) in zip(axs.ravel(), series):
    for c in range(K):
        ax.plot(x[c, ::10], lw=0.4, color=colores[c], alpha=0.8)
    ax.set_title(tit, fontsize=8)
    ax.set_xticks([0, ITER // 20, ITER // 10], ["0", f"{ITER // 2:,}", f"{ITER:,}"])
fig.suptitle("Trazas de las 4 cadenas tras el calentamiento (una de cada 10 iteraciones)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "trazas.png")
plt.close(fig)

# Autocorrelación y densidades por cadena
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
for tit, x in (("m", m1["m"]), ("κ", m1["kappa"]), ("μ (M2)", m2["mu"]), ("σ_δ (M2)", m2["sigma_d"])):
    z = x[0] - x[0].mean()
    ac = np.correlate(z, z, "full")[len(z) - 1:len(z) + 200] / (z.var() * len(z))
    axs[0].plot(ac, lw=1, label=tit)
axs[0].set_title("Autocorrelación (cadena 1)", fontsize=8)
axs[0].legend(fontsize=6)
axs[0].set_xlabel("rezago")
for c in range(K):
    axs[1].hist(m1["kappa"][c], bins=60, histtype="step", density=True, color=colores[c])
    axs[2].hist(m2["sigma_g"][c], bins=60, histtype="step", density=True, color=colores[c])
xs = np.linspace(1, m1["kappa"].max(), 200)
axs[1].plot(xs, st.gamma(2, scale=25).pdf(xs), "k--", lw=0.8, label="prior")
axs[1].set_title("κ: posterior por cadena", fontsize=8)
axs[1].legend(fontsize=6)
axs[2].set_title("σ_γ: posterior por cadena", fontsize=8)
for ax in axs[1:]:
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIG / "autocorr_densidades.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Resultados por celda
# ---------------------------------------------------------------------------
th1 = m1["theta"].reshape(-1, L * F).astype(float)
th2 = m2["theta_new"].reshape(-1, L * F).astype(float)
p1, p2 = th1.mean(0), th2.mean(0)
p_cruda_tr = np.where(n_tr > 0, y_tr / np.maximum(n_tr, 1), y_tr.sum() / n_tr.sum())
p_city = np.full(L * F, y_tr.sum() / n_tr.sum())

R["M1_post"] = dict(m=float(m1["m"].mean()), m_ci=np.quantile(m1["m"], [0.05, 0.95]).tolist(),
                    kappa=float(m1["kappa"].mean()), kappa_ci=np.quantile(m1["kappa"], [0.05, 0.95]).tolist())
R["M2_post"] = {k_: dict(media=float(m2[k_].mean()), ci=np.quantile(m2[k_], [0.05, 0.95]).tolist())
                for k_ in ["mu", "sigma_a", "sigma_b", "sigma_g", "sigma_d"]}
bc = m2["b"].reshape(-1, F)
bc = bc - bc.mean(1, keepdims=True)
R["M2_franja_OR"] = {FRANJAS[f]: dict(OR=float(np.exp(bc[:, f]).mean()),
                                      ci=np.exp(np.quantile(bc[:, f], [0.05, 0.95])).tolist())
                     for f in range(F)}
dc = m2["d"].reshape(-1, 3)
R["M2_anio_d"] = {str(2015 + t): float(dc[:, t].mean()) for t in range(3)}

# grado de seguridad: probabilidad de estar en el top 10 (entre las celdas elegibles)
idx_e = np.where(eleg)[0]


def prob_top10(th):
    sub = th[::10][:, idx_e]
    rk = (-sub).argsort(1).argsort(1)
    pt = np.zeros(L * F)
    pt[idx_e] = (rk < 10).mean(0)
    return pt


pt1, pt2 = prob_top10(th1), prob_top10(th2)
# nivel frente al promedio de la ciudad: P(theta_c > promedio ponderado de la ciudad)
w = n_tr / n_tr.sum()
pmay2 = (th2 > (th2 * w).sum(1, keepdims=True)).mean(0)
nivel = np.where(pmay2 > 0.95, "Alta", np.where(pmay2 < 0.05, "Baja", "Intermedia"))
nivel = np.where(n_tr < 30, "No concluyente", nivel)

tab = c_tr.rename(columns={"n": "n_2015_2017", "y": "graves_2015_2017"}).copy()
tab["p_cruda"] = p_cruda_tr
tab["M1_media"], tab["M1_q05"], tab["M1_q95"] = p1, np.quantile(th1, 0.05, 0), np.quantile(th1, 0.95, 0)
tab["M2_media"], tab["M2_q05"], tab["M2_q95"] = p2, np.quantile(th2, 0.05, 0), np.quantile(th2, 0.95, 0)
tab["M2_P_top10"], tab["M1_P_top10"] = pt2, pt1
tab["M2_P_mayor_ciudad"], tab["nivel"] = pmay2, nivel
tab["n_2018_2019"], tab["graves_2018_2019"] = n_te, y_te
tab["p_obs_2018_2019"] = np.where(n_te > 0, y_te / np.maximum(n_te, 1), np.nan)
tab["elegible"] = eleg
tab["rank_conteo"] = (-y_tr).argsort().argsort() + 1
tab["rank_M2"] = (-p2).argsort().argsort() + 1
tab.to_csv(RES / "celdas_posterior.csv", index=False, encoding="utf-8-sig")
R["niveles"] = tab.nivel.value_counts().to_dict()
R["top10_M2"] = tab.sort_values("M2_media", ascending=False).head(10)[
    ["localidad", "franja", "n_2015_2017", "p_cruda", "M2_media", "M2_q05", "M2_q95", "M2_P_top10",
     "p_obs_2018_2019", "rank_conteo"]].round(3).to_dict("records")
R["top10_conteo"] = tab.sort_values("graves_2015_2017", ascending=False).head(10)[
    ["localidad", "franja", "graves_2015_2017", "M2_media", "rank_M2"]].round(3).to_dict("records")
R["n_top10_M2_en_top10_conteo"] = int(len(set(tab.nsmallest(10, "rank_M2").index)
                                          & set(tab.nsmallest(10, "rank_conteo").index)))
R["n_top10_M2_sur"] = int(np.isin([r["localidad"] for r in R["top10_M2"]], SUR).sum())
R["n_top10_M2_noche"] = int(np.isin([r["franja"] for r in R["top10_M2"]],
                                    ["Madrugada (0-6)", "Noche (18-24)"]).sum())
R["P_top10_alta"] = int((pt2 > 0.9).sum())
R["P_top10_ge_05"] = int((pt2 >= 0.5).sum())
R["P_top10_entre_01_05"] = int(((pt2 >= 0.1) & (pt2 < 0.5)).sum())
R["rho_M2_vs_conteo"] = float(st.spearmanr(p2[eleg], y_tr[eleg])[0])
R["rho_M1_vs_M2"] = float(st.spearmanr(p1[eleg], p2[eleg])[0])
R["ancho_ci_M2_mediana"] = float(np.median(tab.M2_q95 - tab.M2_q05))
R["ancho_ci_M2_peq"] = float(np.median((tab.M2_q95 - tab.M2_q05)[(n_tr < 500) & (n_tr >= 30)]))
R["ancho_ci_M2_grand"] = float(np.median((tab.M2_q95 - tab.M2_q05)[n_tr >= 500]))
R["encogimiento_max_pp_M2"] = float(np.max(np.abs(p2 - p_cruda_tr)[n_tr >= 30]) * 100)

# ---------------------------------------------------------------------------
# 10. Validación con 2018-2019
# ---------------------------------------------------------------------------
val = {"Ciudad (agrupamiento completo)": metricas(p_city),
       "Proporción cruda (sin agrupamiento)": metricas(p_cruda_tr),
       "M1 Beta-Binomial jerárquico": metricas(p1, theta_draws=m1["theta"]),
       "M2 Logística jerárquica": metricas(p2, theta_draws=m2["theta_new"])}
# criterio actual: orden por número de graves (solo orden)
obs_te = np.where(n_te > 0, y_te / np.maximum(n_te, 1), np.nan)
rho_conteo = float(st.spearmanr(y_tr[eleg], obs_te[eleg])[0])
top_c = set(idx_e[np.argsort(-y_tr[idx_e])[:10]])
top_o = set(idx_e[np.argsort(-obs_te[idx_e])[:10]])
val["Conteo de graves (criterio actual)"] = dict(spearman=rho_conteo, top10=len(top_c & top_o))
R["validacion"] = val
for nom in ("M1 Beta-Binomial jerárquico", "M2 Logística jerárquica"):
    v_ = val[nom]
    R.setdefault("criterios", {})[nom] = dict(
        spearman_ok=v_["spearman"] >= 0.90, brier_ok=v_["brier"] <= val["Proporción cruda (sin agrupamiento)"]["brier"],
        top10_ok=v_["top10"] >= 6)

# Recalibración de nivel (solo diagnóstico, no cambia el orden): ¿cuánto del Brier es el cambio de nivel?
shift = sp.logit(R["datos"]["p_te"]) - sp.logit(R["datos"]["p_tr"])
R["brier_M2_nivel_corregido"] = metricas(sp.expit(sp.logit(p2) + shift))["brier"]
R["brier_cruda_nivel_corregido"] = metricas(sp.expit(sp.logit(np.clip(p_cruda_tr, 1e-3, 1 - 1e-3)) + shift))["brier"]

# Figura de encogimiento
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, p_, nom in ((axs[0], p1, "M1"), (axs[1], p2, "M2")):
    s_ = 6 + 60 * np.sqrt(n_tr / n_tr.max())
    ax.scatter(100 * p_cruda_tr[n_tr > 0], 100 * p_[n_tr > 0], s=s_[n_tr > 0], alpha=0.6,
               c=np.where(n_tr[n_tr > 0] < 500, NARANJA, AZUL))
    ax.plot([10, 65], [10, 65], "k:", lw=0.8)
    ax.axhline(100 * p_city[0], color=GRIS, ls="--", lw=0.8)
    ax.set_xlabel("Proporción cruda 2015-2017 (%)")
    ax.set_ylabel(f"Media posterior {nom} (%)")
    ax.set_title(f"{nom}: las celdas pequeñas (naranja) se acercan a la ciudad", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "encogimiento.png")
plt.close(fig)

# Forest plot del top 20 de M2
top20 = tab.sort_values("M2_media", ascending=False).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
yy = np.arange(len(top20))
ax.errorbar(100 * top20.M2_media, yy, xerr=[100 * (top20.M2_media - top20.M2_q05), 100 * (top20.M2_q95 - top20.M2_media)],
            fmt="o", color=AZUL, ms=4, lw=1, label="M2: media e intervalo creíble 90 %")
ax.scatter(100 * top20.p_obs_2018_2019, yy, marker="x", color=ROJO, s=20, zorder=3, label="Observado 2018-2019")
ax.set_yticks(yy, [f"{a} | {b.split(' ')[0]}  (P top10={p:.2f}; #{r} en conteo)"
                   for a, b, p, r in zip(top20.localidad, top20.franja, top20.M2_P_top10, top20.rank_conteo)], fontsize=6.5)
ax.axvline(100 * R["datos"]["p_tr"], color=GRIS, ls="--", lw=0.8)
ax.set_xlabel("% de siniestros graves")
ax.legend(fontsize=7, loc="upper left")
ax.set_xlim(22, 64)
ax.set_title("Top 20 de M2 (2015-2017) frente a lo observado en 2018-2019", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "top20_forest.png")
plt.close(fig)

# Figura de validación: predicción vs observado y orden por conteo
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
e = eleg
axs[0].scatter(100 * p2[e], 100 * obs_te[e], s=12, color=AZUL, label=f"M2 (ρ = {val['M2 Logística jerárquica']['spearman']:.2f})")
axs[0].plot([15, 60], [15, 60], "k:", lw=0.8)
axs[0].plot([15, 60], [15 + 100 * (R['datos']['p_te'] - R['datos']['p_tr']), 60 + 100 * (R['datos']['p_te'] - R['datos']['p_tr'])],
            color=GRIS, ls="--", lw=0.8, label="cambio de nivel de la ciudad")
axs[0].set_xlabel("Predicción M2 con 2015-2017 (%)")
axs[0].set_ylabel("Observado 2018-2019 (%)")
axs[0].legend(fontsize=7)
axs[1].scatter(y_tr[e], 100 * obs_te[e], s=12, color=NARANJA, label=f"conteo (ρ = {rho_conteo:.2f})")
axs[1].set_xlabel("Siniestros graves 2015-2017 (criterio actual)")
axs[1].set_ylabel("Observado 2018-2019 (%)")
axs[1].legend(fontsize=7)
fig.suptitle(f"Validación en {e.sum()} celdas con al menos 30 siniestros en ambos periodos", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "validacion.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Cinco sensibilidades fijadas en el Avance 2
# ---------------------------------------------------------------------------
sens = {}
for nom, rt in (("kappa centrada en 10", 2 / 10), ("kappa centrada en 300", 2 / 300)):
    o = correr_m1(n_tr, y_tr, rt_k=rt, seed=SEED + 11)
    mt = metricas(o["theta"].reshape(-1, L * F).mean(0))
    sens[nom] = dict(spearman=mt["spearman"], brier=mt["brier"], top10=mt["top10"],
                     kappa=float(o["kappa"].mean()),
                     rho_vs_M1=float(st.spearmanr(o["theta"].reshape(-1, L * F).mean(0)[eleg], p1[eleg])[0]))
# franjas de tres horas
c3_tr = celdas(tr, "franja3h", FR3)
c3_te = celdas(te, "franja3h", FR3)
o = correr_m1(c3_tr.n.values, c3_tr.y.values, seed=SEED + 12)
p3 = o["theta"].reshape(-1, len(c3_tr)).mean(0)
el3 = (c3_tr.n.values >= 30) & (c3_te.n.values >= 30)
ob3 = np.where(c3_te.n > 0, c3_te.y / np.maximum(c3_te.n, 1), np.nan)
cr3 = np.where(c3_tr.n > 0, c3_tr.y / np.maximum(c3_tr.n, 1), 0.33)
i3 = np.where(el3)[0]
sens["franjas de 3 horas (160 celdas)"] = dict(
    spearman=float(st.spearmanr(p3[el3], ob3[el3])[0]),
    spearman_cruda=float(st.spearmanr(cr3[el3], ob3[el3])[0]),
    top10=len(set(i3[np.argsort(-p3[i3])[:10]]) & set(i3[np.argsort(-ob3[i3])[:10]])),
    celdas_eleg=int(el3.sum()),
    brier=float(metricas(p3, c3_te.n.values, c3_te.y.values, el3)["brier"]),
    brier_cruda=float(metricas(cr3, c3_te.n.values, c3_te.y.values, el3)["brier"]))
# solo siniestros con muertos (menos sensible al subregistro de choques leves)
cm_tr, cm_te = celdas(tr, y="con_muertos"), celdas(te, y="con_muertos")
o = correr_m1(cm_tr.n.values, cm_tr.y.values, a_m=1.0, b_m=30.0, seed=SEED + 13)
pm = o["theta"].reshape(-1, L * F).mean(0)
obm = np.where(cm_te.n > 0, cm_te.y / np.maximum(cm_te.n, 1), np.nan)
sens["solo siniestros con muertos"] = dict(
    spearman=float(st.spearmanr(pm[eleg], obm[eleg])[0]),
    spearman_cruda=float(st.spearmanr(np.where(cm_tr.n > 0, cm_tr.y / np.maximum(cm_tr.n, 1), 0)[eleg], obm[eleg])[0]),
    rho_vs_graves=float(st.spearmanr(pm[eleg], p2[eleg])[0]),
    tasa_muertos_tr=float(cm_tr.y.sum() / cm_tr.n.sum()),
    m=float(o["m"].mean()), kappa=float(o["kappa"].mean()),
    top10_comun_con_M2=len(set(idx_e[np.argsort(-pm[idx_e])[:10]]) & set(idx_e[np.argsort(-p2[idx_e])[:10]])),
    top10=[f"{c_tr.localidad[i]} | {c_tr.franja[i]} ({100 * pm[i]:.1f} %)" for i in idx_e[np.argsort(-pm[idx_e])[:10]]],
    top10_sur=int(es_sur[idx_e[np.argsort(-pm[idx_e])[:10]]].sum()),
    top10_noche=int(es_noche[idx_e[np.argsort(-pm[idx_e])[:10]]].sum()))
# 2020 (pandemia): ¿se sostiene el orden?
n20, y20 = c_20.n.values, c_20.y.values
el20 = (n_tr >= 30) & (n20 >= 30)
ob20 = np.where(n20 > 0, y20 / np.maximum(n20, 1), np.nan)
i20 = np.where(el20)[0]
sens["2020 (ruptura)"] = dict(
    spearman_M2=float(st.spearmanr(p2[el20], ob20[el20])[0]),
    spearman_cruda=float(st.spearmanr(p_cruda_tr[el20], ob20[el20])[0]),
    spearman_conteo=float(st.spearmanr(y_tr[el20], ob20[el20])[0]),
    top10_M2=len(set(i20[np.argsort(-p2[i20])[:10]]) & set(i20[np.argsort(-ob20[i20])[:10]])),
    celdas_eleg=int(el20.sum()))
R["sensibilidades"] = sens

# ---------------------------------------------------------------------------
# 12. Guardar
# ---------------------------------------------------------------------------
pd.DataFrame(val).T.to_csv(RES / "validacion_2018_2019.csv", encoding="utf-8-sig")


def limpio(o):
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpio(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(RES / "resumen_resultados.json", "w", encoding="utf-8") as fh:
    json.dump(limpio(R), fh, ensure_ascii=False, indent=1)

# Salida corta en consola
print(f"Tiempo MCMC: {R['tiempo_s']}")
print("Diagnósticos:", json.dumps(limpio(R["diag"]), ensure_ascii=False))
print(pd.DataFrame(val).T.round(4).to_string())
print("Criterios:", R["criterios"])
