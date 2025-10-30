from __future__ import annotations

import unittest

from ctr25.cli import INDUSTRY_SIMPLIFY_MAP
from ctr25.utils.names import normalize_company_name
from ctr25.utils.text import classify_industry, load_industry_map


def _classify(value: str, **context):
    imap = load_industry_map()
    slug, details = classify_industry((value, context), imap)
    simplified = INDUSTRY_SIMPLIFY_MAP.get(slug, slug)
    return slug, simplified, details


class IndustryNormalizationTests(unittest.TestCase):
    def test_biotech_company_maps_to_healthcare(self):
        slug, simplified, details = _classify(
            "industrial_manufacturing",
            company_name="INOVACIONES INDUSTRIALES BIOTECNOLÓGICAS S.A. DE C.V.",
            source_meta="sector: Biotechnology",
        )
        self.assertEqual("healthcare_pharma_biotech", simplified)
        self.assertEqual("alias", details["reason"])

    def test_jumbo_chile_reclassified_to_retail(self):
        slug, simplified, details = _classify(
            "agro_food_beverage",
            company_name="Jumbo Chile",
            text_snippet="El supermercado Jumbo Chile lanzó una nueva campaña verde.",
        )
        self.assertEqual("retail_consumer", simplified)
        self.assertIn(slug, {"retail_consumer_services", "retail_consumer"})

    def test_del_pinar_name_normalization(self):
        normalized = normalize_company_name(
            {
                "company_name": "Del Pinar Sociedad Anónima",
                "text_snippet": "Del Pilar Sociedad Anónima anuncia inversiones.",
            }
        )
        self.assertEqual("Del Pilar Sociedad Anónima", normalized)

    def test_instituto_videira_maps_to_public_sector(self):
        slug, simplified, details = _classify(
            "retail_consumer_services",
            company_name="Instituto Vida videira",
            text_snippet="El instituto Vida Videira amplía su programa educativo.",
        )
        self.assertEqual("public_social_education", simplified)
        self.assertIn(details["reason"], {"alias", "weighted"})


if __name__ == "__main__":
    unittest.main()
