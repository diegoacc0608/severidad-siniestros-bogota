# ¿Dónde y cuándo son más graves los siniestros viales en Bogotá?

Severidad relativa de los siniestros viales por localidad y franja horaria (2015–2019) con modelos jerárquicos bayesianos.

**Grupo 1** · Estadística Bayesiana y Pensamiento Crítico en Ciencia de Datos 2 · Pregrado en Ciencia de Datos, Universidad Externado de Colombia, 2026

Integrantes: Jean Paul Rojas Rojas, Daniel Estiven Pacheco Villagrande y Diego Andrés Cascante Camacho.

Repositorio: https://github.com/diegoacc0608/severidad-siniestros-bogota

## De qué trata

Ordenar las zonas de Bogotá por el número de siniestros que acumulan muestra dónde se concentran, pero no dónde son más graves. Estimamos, para las 80 combinaciones de localidad y franja horaria, la probabilidad de que un siniestro registrado sea grave (con heridos o muertos) y su incertidumbre. Usamos dos modelos jerárquicos bayesianos ajustados con MCMC programado a mano:

- **M1:** Beta-Binomial jerárquico (Gibbs + Metropolis-Hastings).
- **M2:** regresión logística jerárquica con efectos de localidad, franja, interacción y año (Metropolis-Hastings por componentes + Gibbs).

Los modelos aprenden con 2015–2017 y se validan una sola vez con 2018–2019. Es un prototipo retrospectivo: estima severidad registrada, no riesgo por exposición.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `Grupo1_EntregaFinal_Bayesiana.pdf` | Artículo final |
| `Grupo1_EntregaFinal_Bayesiana.tex`, `Bibliografia.bib`, `apalike_sf.bst` | Fuente LaTeX del artículo |
| `Grupo1_EntregaFinal_Bayesiana.py` | Script que reproduce todas las cifras y figuras del artículo |
| `figuras/` | Las 10 figuras del artículo |
| `resultados/` | Resumen de resultados, diagnósticos de convergencia, tabla por celda y validación |
| `notebooks/Grupo1_Bayesiana_completo.ipynb` | Notebook con todo el análisis, ejecutable en Colab |
| `notebooks/por_etapas/` | El mismo análisis en ocho notebooks (01 a 08) |
| `notebooks/comun.py`, `notebooks/mcmc.py` | Código común y muestreadores MCMC |
| `notebooks/datos/` | Base limpia y tablas de celdas |
| `notebooks/salidas/` | Figuras, tablas y resultados que producen los notebooks |

`notebooks/README.md` indica qué celda produce cada cifra y figura del artículo.

## Cómo reproducirlo

```bash
pip install numpy pandas scipy matplotlib arviz==0.23.4
python Grupo1_EntregaFinal_Bayesiana.py
```

El script busca `siniestros_limpios.csv` junto a él o en `datos/`; para correrlo desde la raíz del repositorio, copie `notebooks/datos/siniestros_limpios.csv` a `datos/`. Semilla: 20261109. Versiones con las que se ejecutó: Python 3.12.10, NumPy 2.3.5, pandas 2.3.3, SciPy 1.18.1, Matplotlib 3.10.7 y ArviZ 0.23.4. Con ArviZ 1.x la función `az.from_dict` cambió y el script no corre.

## Datos

Fuente: Secretaría Distrital de Movilidad, *Siniestros viales consolidados Bogotá D.C.* (2015–2020), portal Datos Abiertos Bogotá: https://datosabiertos.bogota.gov.co/dataset/siniestros-viales-consolidados-bogota-d-c

Trabajamos con una copia descargada por el equipo. Al volver a consultar el portal, el 9 de noviembre de 2026, el recurso aparecía como restringido; no conocemos el motivo. El Excel original (47 MB) no se incluye por su tamaño; sí se incluye la base limpia que se deriva de él (`notebooks/datos/siniestros_limpios.csv`), que no contiene datos personales. La sección 3.1 del artículo explica la procedencia con más detalle.
