# Climate Tech Report 2025 (Emergente Analytics)


**Propósito**: detectar, medir y priorizar la demanda corporativa de soluciones climáticas en LATAM, con un índice representativo (IIC) y un Prospect Score (PS) accionable para BD.


## Alcance
- Países: MX, BR, CO, CL, AR, UY (ampliable)
- Industrias: Energía/Utilities, Agro/Alimentos, Construcción/Real Estate, Retail/Consumo, Manufactura
- Tamaño: 50–249, 250–999, 1000+
- Ventana: últimos 12 meses


## Quickstart
```bash
# 0) clonar e instalar (entorno virtual recomendado)
pip install -e .


# 1) inicializar carpeta y configs
ctr25 init


# 2) construir marco muestral MOCK (o integrar fuentes reales)
ctr25 sample --use-mock


# 3) recolectar señales (demo) → data/processed/events_normalized.csv
ctr25 collect --demo


# 4) computar IIC por empresa (estratificado y normalizado)
ctr25 compute-iic


# 5) calcular Prospect Score y etiquetas
ctr25 compute-ps


# 6) visualizar (heatmap país×industria + tabla de cuentas)
ctr25 viz


# 7) exportar listas para colaboradores (CSV)
ctr25 export --out reports/collaborators/prospects.csv