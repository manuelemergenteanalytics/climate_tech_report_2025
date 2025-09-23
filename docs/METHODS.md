# Metodología


## 1) Universo y muestra
- Estratificación: País × Industria × Tamaño.
- Muestra aleatoria por estrato; pesos = N_estrato / n_estrato.
- Ventana: últimos 12 meses.
- Reporte: IC95% por estrato (bootstrapping estratificado).


## 2) Señales
- **H Hiring/Org**: vacantes ESG por 1.000 empleados; existencia CSO/comité.
- **C Compromisos**: SBTi/RE100/ISO/B Corp (ponderados por antigüedad/alcance).
- **F Finanzas**: bonos verdes/préstamos; montos relativos si disponible.
- **P Proyectos**: pilotos/alianzas verificadas (12m).
- **X Comunicación**: menciones en newsroom/prensa (deduplicadas; normalizadas por 1.000 palabras).


## 3) IIC (0–100)
- Cálculo de z-score intra-estrato por pilar; winsorización p95; rebasing 0–100.
- IIC = 0.25·H + 0.20·C + 0.20·F + 0.20·P + 0.15·X.


## 4) Prospect Score (PS)
- PS = α·IIC + β·Señales calientes (recencia 90d; densidad de señales), con etiquetas.
- Reglas ejemplo: Empleo ESG activo + SBTi + prensa 90d ⇒ "Muy Alto".


## 5) QA & Sesgos
- Silencios = 0 (se incluyen).
- Normalización por tamaño/industria.
- Deduplicación de prensa y ventana fija 12m.
- Validación manual 10–15% para calibración de pesos/regex.