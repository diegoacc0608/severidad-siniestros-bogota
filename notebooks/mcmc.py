"""Muestreadores MCMC programados a mano (M1 y M2), idénticos a los de los notebooks 05 y 06
y a Grupo2_EntregaFinal_Bayesiana.py. El notebook 08 los importa para las sensibilidades."""
import numpy as np
from scipy import special as sp
import arviz as az
from comun import K, BURN, ITER, SEED, L, F

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
