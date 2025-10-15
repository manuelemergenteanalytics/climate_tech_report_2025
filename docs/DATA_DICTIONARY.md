# Diccionario de datos (principal)


## events_normalized.csv (nivel: señal)
- company_id, company_name, country, industry, size_bin
- source (jobs, memberships, news, webscan, finance)
- signal_type (e.g., job_posting, sbti, re100, green_bond, pilot_news)
- signal_strength (float)
- ts (UTC ISO)
- url, title, text_snippet
- description (resumen generado por LLM/regex con evidencias del evento)


## company_scores.csv (nivel: empresa)
- company_id, company_name, country, industry, size_bin, weight_stratum
- H, C, F, P, X (z-score intra-estrato)
- IIC (0–100)
- hot_signals_90d, hot_score
- PS (0–100), label {Muy Alto/Alto/Medio}
