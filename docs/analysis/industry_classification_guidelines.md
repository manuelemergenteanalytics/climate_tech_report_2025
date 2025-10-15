# Industry Classification Guidelines

## Target Taxonomy (15 labels)
| slug | human label | scope & signals | indicative keywords / phrases |
| --- | --- | --- | --- |
| energy_power_utilities | Energy & Utilities | Generation, transmission, distribution of electricity, renewables, water or waste utilities | energy, utility, power, solar, wind, hydro, grid, renewables, transmission, distribution |
| oil_gas | Oil & Gas | Exploration, production, transport or refining of hydrocarbons and derivatives | oil, petro, gas, upstream, downstream, refinery, lng, pipeline |
| mining_materials | Mining & Basic Materials | Extraction/processing of metals, minerals, basic chemicals, pulp & paper | mining, mineral, steel, metal, smelter, pulp, paper, forest products, quarry |
| industrial_manufacturing | Industrial Manufacturing | Production of machinery, equipment, auto, electronics, packaging, industrial goods | manufacturing, factory, equipment, machinery, automotive, electronics, components |
| construction_infrastructure_realestate | Construction & Real Assets | Construction, engineering, infrastructure, real estate development/management | construction, engineering, infrastructure, real estate, property, builders, contractors |
| transport_mobility_logistics | Transport & Logistics | Passenger/freight transport, logistics, airlines, shipping, ports, rail | transport, logistics, mobility, trucking, rail, airline, shipping, fleet, ports |
| agro_food_beverage | Agriculture & Food | Agriculture, fishing, food & beverage production/supply chains | agriculture, agro, food, beverage, farming, fisheries, crop, livestock, dairy |
| retail_consumer_services | Consumer & Retail | Retailers, consumer goods brands, e-commerce, hospitality-lite services to consumers | retail, consumer goods, e-commerce, stores, wholesale, distribution, apparel |
| finance_insurance_capital | Finance & Insurance | Banking, insurance, investment, capital markets, fintech | bank, insurance, financial services, asset management, investment, fintech, capital |
| ict_digital_media | ICT, Digital & Media | Software, IT services, telecom, digital platforms, media, data centers | software, technology, telecom, digital, telecom, media, cloud, data center, platform |
| healthcare_pharma_biotech | Health & Life Sciences | Healthcare providers, hospitals, pharma, biotech, medical devices | healthcare, hospital, pharma, biotech, medical, clinic, health services |
| professional_services_consulting | Professional & Technical Services | Consulting, legal, accounting, engineering/advisory services | consulting, advisory, legal, accounting, professional services, design studios |
| hospitality_tourism_leisure | Hospitality & Leisure | Accommodation, restaurants, travel, recreation, entertainment venues | hotel, hospitality, tourism, travel, leisure, restaurants, resorts, entertainment |
| environmental_circular_services | Environmental & Circular Economy | Waste management, recycling, carbon services, environmental consulting | waste, recycling, circular, environmental services, carbon markets, remediation |
| public_social_education | Public, Social & Education | Public sector entities, NGOs, education, research institutions, cooperatives | government, municipal, public agency, ngo, nonprofit, cooperative, university, school |

## Column Priority by Source
- **SBTi (`sbti_data.csv`)**: 1) `sector`, 2) `member_name`, 3) external enrichment via news snippets if available.
- **UNGC (`ungc_participants_latam.csv`)**: 1) `sector` (token classification), 2) `type` when it encodes SMEs vs companies (size only), 3) `name` fallback for state/ngo cues, 4) `status` text if informative.
- **B Corp (`b_corp_data.csv`)**: 1) `industry_category`, 2) tokens from `description`, 3) `products_and_services` when present.
- **News (`data/raw/news/**/*`)**: 1) existing `industry` when reliable, 2) keywords within `title` + `text_snippet`, 3) previously assigned industry for the same `company_id` in other sources.

## Keyword Weighting & Matching
- Convert text to lower case, strip accents, and split on non-alphanumeric boundaries.
- Use curated keyword lexicons per label (to be stored in `config/industry_map.yml`), including language variants (ES/PT/EN) and common misspellings.
- Give precedence to multi-word matches (e.g., "power generation" should outweigh a single token like "power").
- Penalize generic words by requiring at least two positive hits before assigning highly generic labels such as `professional_services_consulting`.
- When multiple labels match, resolve by scoring: `score(label) = 2 * multi_word_hits + 1 * single_word_hits + source_weight`. Source weights: SBTi=3, BCorp=2, UNGC=1.5, News=1.

## Conflict Resolution
- If two labels have scores within 1 point, pick the one with higher-priority source (SBTi > B Corp > UNGC > News). If tie remains, prefer the label already assigned to the company in `events_normalized.csv` to maintain continuity.
- When no label reaches minimum score (>=2), default to `professional_services_consulting` **only** if the text explicitly mentions consulting, advisory, or services; otherwise mark as `public_social_education` (for NGOs/municipal) or `industrial_manufacturing` (for generic "industrial" references) based on context.
- Repeated signals from the same company and source diminish incremental weight by 0.5 per repetition (e.g., second identical news article contributes half the weight for tie-breaking).

## Description Synthesis Rules
- Aggregate descriptive fields (prefer order: B Corp description > SBTi sector > UNGC status > News snippet).
- Deduplicate sentences and keep maximum 400 characters per event.
- Include leading clause specifying source and industry, e.g., `"SBTi sector: Software and Services. B Corp focus: certified organic food retailer."`
- For companies with multiple sources, merge sentences separated by `" | "` and remove redundant adjectives via post-processing regex (e.g., collapse repeated "sustainable").

## Documentation & Storage
- Keep this taxonomy in sync with `config/industry_map.yml`; document any future changes in `docs/analysis/industry_classification_guidelines.md`.
- Store intermediate LLM outputs (raw classification prompts/responses) in `data/interim/industry_llm/` with filenames `<source>_<yyyymmdd>.jsonl` for auditing.
- Version-control generated summaries by writing deterministic pipelines (no timestamps inside the output content).

## Quality Checks
- Minimum acceptance: less than 8% of rows in `events_normalized.csv` should fall into fallback buckets after reclassification.
- Maintain counts by label across sources (produce a summary CSV in `data/processed/industry_distribution.csv`).
- Randomly sample 20 events per release and verify the assigned industry + description manually (document samples in `docs/analysis/industry_spotchecks.md`).
