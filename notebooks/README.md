# Notebooks del Grupo 2 — Estadística Bayesiana, entrega final

Severidad relativa de los siniestros viales en Bogotá por localidad y franja horaria (2015–2019).
Integrantes: Jean Paul Rojas Rojas, Daniel Estiven Pacheco Villagrande, Diego Andrés Cascante Camacho.

## El notebook principal: `Grupo2_Bayesiana_completo.ipynb`

Un solo notebook reúne todo el trabajo, de la base de la Secretaría Distrital de Movilidad (SDM) a cada cifra y figura del informe `Grupo2_EntregaFinal_Bayesiana.pdf`. Ya trae sus salidas guardadas y se corre de principio a fin: en Colab, **Entorno de ejecución → Ejecutar todas**.

| Sección | Qué hace | Empieza en la celda |
|---|---|---|
| 0. Configuración y código común | Monta el Drive, escribe `comun.py` y `mcmc.py` e imprime versiones y semilla | 2 |
| 1. Carga y limpieza | Del Excel de la SDM a `datos/siniestros_limpios.csv` (secciones 1–2 del EDA original) | 8 |
| 2. Análisis exploratorio descriptivo | EDA 2015–2019 (sección 3 del EDA original) y Figuras 1–4 del informe | 33 |
| 3. Del EDA al modelo | Forma por celda, χ², estructura, priors de la Primera Entrega, partición (secciones 4–6 del EDA original) | 62 |
| 4. Priors y chequeo previo | Tabla 1 y Figura 5 | 80 |
| 5. M1 Beta-Binomial (Gibbs + MH) | Algoritmo, corrida, R̂ y ESS | 84 |
| 6. M2 logística jerárquica (MH + Gibbs) | Algoritmo, corrida, R̂ y ESS | 91 |
| 7. Convergencia, resultados y validación 2018–2019 | Figuras 6–10, Tabla 2, equidad, capacidad y verificación contra el informe | 98 |
| 8. Sensibilidades | Tabla 3 | 106 |
| 9. Comprobación final | Cifras clave y figuras idénticas a las del informe | 109 |

La numeración de celdas cuenta desde 1, incluidas las de texto.

## Cómo correrlo en Google Colab

1. Suba la carpeta `notebooks` completa a su Google Drive.
2. Abra `Grupo2_Bayesiana_completo.ipynb` con Colab y ejecute todo. La primera celda de código monta el Drive y busca la carpeta sola; si no la encuentra, escriba su ruta en `BASE_COLAB`.
3. La sección 1 necesita el Excel de la SDM en `notebooks/datos/`. No viene incluido porque pesa 47 MB. Sin él, ejecute la sección 0 y siga desde la 2: `datos/siniestros_limpios.csv` sí viene.

Tiempo total en un portátil: unos 4 minutos, de los cuales 2,5 son la lectura del Excel.

## De dónde sale cada cifra y figura del informe

| Cifra o figura del informe | Sección y celda del notebook principal |
|---|---|
| 196.152 siniestros; 173.443 en 2015–2019, 59.303 graves (34,2 %), 2.626 con muertos | Sección 1; sección 2, apartado 3.1 |
| 175 con gravedad contradictoria, 24 dudosos, 38,9 % de horas en punto o y media | Sección 1, apartados 2.3 y 2.5 |
| `siniestros_limpios.csv` idéntico a la copia del equipo | Sección 1, última celda (control de trazabilidad) |
| Día de la semana: 33,0 % los viernes frente a 37,3 % los domingos | Sección 2, apartado 3.6 |
| Spearman localidad −0,63; Figuras 1–4 (`eda_hora`, `eda_localidad`, `eda_celdas`, `eda_anio`) | Sección 2, última celda |
| χ² = 4.648 con 74 g. l.; localidad + franja explican el 94,9 % | Sección 3, apartados 4.2 y 4.3 |
| 80 celdas, 77 con datos, 75 evaluables, 21 pequeñas; 101.500 / 71.943 siniestros | Sección 3, última celda |
| Tabla 1 (priors) y Figura 5 (chequeo a priori; percentiles 20–83) | Sección 4 |
| M1: m = 0,367, κ = 23,5, aceptación 32 %, R̂ = 1,00, ESS ≥ 8.961 | Sección 5 |
| M2: σ de localidad, franja, interacción y año; razones de odds por franja; R̂ = 1,00, ESS ≥ 9.204 | Sección 6 |
| Figuras 6 y 7 (trazas, autocorrelación) | Celda 100 |
| Rango 17,6 %–54,7 %; niveles 41/10/24/5; prioridad 1 = 9 celdas, prioridad 2 = 32; ρ M2 frente a conteo −0,19 | Celda 101 |
| Tabla 2: Spearman 0,933, Brier 0,22169, top 10 6/10, cobertura 94,9 %, lpd −375; criterios de éxito | Celda 102 |
| Equidad: celdas pequeñas 6,69 frente a 3,45 pp, piso binomial 3,0 / 1,3; capacidad 6.048 (52,4 %) frente a 21.016 (36,9 %) | Celda 103 |
| Figuras 8, 9 y 10 (encogimiento, validación, top 20) | Celda 104 |
| Verificación automática contra el informe (falla si algo no coincide) | Celda 105 |
| Tabla 3: κ = 10 y 300, franjas de 3 h, solo muertos, 2020 | Celda 108 |
| Comprobación de las figuras frente al informe: 7 idénticas byte a byte y 3 iguales a la vista (difieren solo en el suavizado de bordes, en menos del 0,3 % de los píxeles) | Celda 110 |

## Carpetas

- `datos/`: `siniestros_limpios.csv` y las tablas de celdas. Aquí va también el Excel de la SDM si se quiere correr la sección 1.
- `salidas/figuras/`: las 10 figuras del informe. `salidas/resultados/`: tablas, diagnósticos y un `.json` por etapa. `salidas/eda_original/`: figuras y tablas del EDA original. `salidas/m1_resumen.npz` y `m2_resumen.npz`: resúmenes de las cadenas que usan las secciones 7 y 8.
- `comun.py` y `mcmc.py`: los escribe la sección 0 del notebook principal. No hay que editarlos.
- `por_etapas/`: respaldo con las mismas ocho etapas en notebooks separados (`01_carga_y_limpieza` … `08_sensibilidades`), también ejecutados. Sirve para revisar o correr una sola etapa; producen las mismas salidas.
- `../Grupo2_EntregaFinal_Bayesiana.py`: el mismo análisis en un solo script, sin el EDA original.

## Datos de origen

- **Fuente:** Secretaría Distrital de Movilidad, *Siniestros viales consolidados Bogotá D.C. (2015–2020)*, publicada en Datos Abiertos Bogotá con licencia CC BY 4.0.
- **Copia usada:** la que descargó el equipo del portal Datos Abiertos Bogotá (https://datosabiertos.bogota.gov.co/dataset/siniestros-viales-consolidados-bogota-d-c). La copia más antigua que conservamos es del 12 de agosto de 2026 y la usada para correr el código, del 25 de octubre de 2026; ambas tienen las mismas filas por hoja. Según sus propiedades internas, el archivo se creó el 8 de abril de 2021 y se modificó por última vez el 21 de octubre de 2021. El 9 de noviembre de 2026 el recurso aparecía como restringido en el portal; no conocemos el motivo y no le atribuimos una causa.
- **Capa vigente de la SDM (no usada en este trabajo):** *Siniestralidad BD*, Portal de Datos de la SDM, con siniestros georreferenciados desde 2007 y actualización diaria: https://datos.movilidadbogota.gov.co/datasets/ea243e7de8e846c8bd27e47c08771d66

## Reproducibilidad

- **Semilla del modelo final:** 20261109 (M1, 20261109; M2, 20261110; chequeo a priori, 20261116; sensibilidades, 20261120 a 20261122).
- **Semilla de las partes heredadas del EDA original** (secciones 2 y 3): 2026, la misma de la Primera Entrega.
- **MCMC:** 4 cadenas, 5.000 iteraciones de calentamiento y 20.000 guardadas por cadena.
- **Versiones con las que se ejecutó:** Python 3.12.10, numpy 2.3.5, pandas 2.3.3, scipy 1.18.1, matplotlib 3.10.7, arviz 0.23.4, seaborn 0.13.2, holidays 0.108. La sección 0 imprime las versiones con que se corre.
