from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import yaml

from ctr25.utils.text import clean_text

DEFAULT_MODEL = os.getenv("INDUSTRY_LLM_MODEL", "gpt-4o-mini")
DEFAULT_PROVIDER = os.getenv("INDUSTRY_LLM_PROVIDER", "openai")


@dataclass
class ClassificationOutput:
    industry_slug: str
    confidence: float
    description: str
    provider: str
    model: str
    cached: bool = False
    used_fallback: bool = False


class IndustryPatternMatcher:
    """Regex-based scorer that mirrors config/industry_map.yml."""

    def __init__(self, map_path: str = "config/industry_map.yml") -> None:
        self.map_path = Path(map_path)
        self.patterns: List[Tuple[re.Pattern[str], str, float]] = []
        self._labels: List[str] = []
        self._load()

    def _load(self) -> None:
        if not self.map_path.exists():
            raise FileNotFoundError(
                f"Industry map not found at {self.map_path}. Please create it before classification."
            )
        raw = yaml.safe_load(self.map_path.read_text(encoding="utf-8")) or {}
        mappings = raw.get("mappings") or []
        labels: List[str] = []
        patterns: List[Tuple[re.Pattern[str], str, float]] = []
        for entry in mappings:
            target = entry.get("to")
            pattern = entry.get("pattern")
            weight = float(entry.get("weight", 1.0))
            if not target or not pattern:
                continue
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
            patterns.append((compiled, str(target), weight))
            labels.append(str(target))
        self.patterns = patterns
        self._labels = sorted(set(labels))

    @property
    def labels(self) -> Sequence[str]:
        return tuple(self._labels)

    def score(self, texts: Sequence[str]) -> Tuple[str, Dict[str, float]]:
        bucket: Dict[str, float] = {label: 0.0 for label in self._labels}
        corpus = " \n ".join(clean_text(text or "") for text in texts if isinstance(text, str))
        if not corpus:
            return "", bucket
        for regex, target, weight in self.patterns:
            if regex.search(corpus):
                bucket[target] = bucket.get(target, 0.0) + weight
        best_label = max(bucket.items(), key=lambda item: item[1])[0] if bucket else ""
        return best_label, bucket


class IndustryLLMClassifier:
    """LLM-backed classifier with regex fallback."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
        cache_dir: str | Path = "data/interim/industry_llm",
        map_path: str = "config/industry_map.yml",
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.provider = provider or DEFAULT_PROVIDER
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE") or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
        self.matcher = IndustryPatternMatcher(map_path)
        self.enabled = bool(self.api_key)

    def classify(self, payload: Dict[str, object]) -> ClassificationOutput:
        cache_key = self._cache_key(payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return ClassificationOutput(
                industry_slug=data.get("industry_slug", ""),
                confidence=float(data.get("confidence", 0.5)),
                description=data.get("description", ""),
                provider=data.get("provider", self.provider),
                model=data.get("model", self.model),
                cached=True,
                used_fallback=bool(data.get("used_fallback", False)),
            )

        result: Optional[ClassificationOutput] = None
        raw_response: Optional[Dict[str, object]] = None

        if self.enabled:
            try:
                raw_response = self._invoke_llm(payload)
                parsed = self._parse_response(raw_response)
                if parsed:
                    result = ClassificationOutput(
                        industry_slug=parsed.get("industry_slug", ""),
                        confidence=float(parsed.get("confidence", 0.6)),
                        description=parsed.get("description", ""),
                        provider=self.provider,
                        model=self.model,
                        cached=False,
                        used_fallback=False,
                    )
            except Exception:
                result = None

        if result is None or not result.industry_slug:
            fallback = self._fallback(payload)
            result = ClassificationOutput(**fallback)
            result.used_fallback = True

        cache_payload = {
            "industry_slug": result.industry_slug,
            "confidence": result.confidence,
            "description": result.description,
            "provider": result.provider,
            "model": result.model,
            "used_fallback": result.used_fallback,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "input": payload,
        }
        if raw_response is not None:
            cache_payload["raw_response"] = raw_response
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _cache_key(self, payload: Dict[str, object]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _invoke_llm(self, payload: Dict[str, object]) -> Dict[str, object]:
        system_prompt = (
            "You are an analyst that maps Latin American companies to industry tags. "
            "Respond with compact JSON only."
        )
        allowed_labels = ", ".join(self.matcher.labels)
        field_lines = []
        for key, value in payload.get("fields", {}).items():
            if not value:
                continue
            field_lines.append(f"- {key}: {value}")
        context = "\n".join(field_lines) or "- (no additional context)"
        user_prompt = textwrap.dedent(
            f"""
            Company: {payload.get('company_name', '')}
            Source: {payload.get('source', '')}
            Country: {payload.get('country', '')}
            Existing industry hints: {payload.get('hints', '') or 'none'}
            Details:\n{context}

            Choose the single best industry slug from: [{allowed_labels}].
            If unsure pick the closest fit.
            Reply with JSON matching:
            {{
              "industry_slug": "one_of_allowed_slugs",
              "confidence": float_between_0_and_1,
              "description": "<=400 chars summary combining the most relevant facts"
            }}
            """
        ).strip()
        endpoint = self._chat_endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        response = requests.post(endpoint, headers=headers, json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    def _chat_endpoint(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        base = self.api_base.rstrip("/")
        return f"{base}/chat/completions"

    def _parse_response(self, response: Dict[str, object]) -> Optional[Dict[str, object]]:
        choices = response.get("choices")
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content")
        if not content:
            return None
        json_str = self._extract_json(content)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        industry_slug = data.get("industry_slug", "")
        if industry_slug and industry_slug not in self.matcher.labels:
            # Attempt to normalize to known labels
            industry_slug = industry_slug.strip().lower()
            best, _ = self.matcher.score([industry_slug])
            data["industry_slug"] = best or ""
        return data

    def _extract_json(self, text: str) -> Optional[str]:
        matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        if not matches:
            return None
        # choose the first JSON-looking block
        return matches[0]

    def _fallback(self, payload: Dict[str, object]) -> Dict[str, object]:
        texts: List[str] = []
        fields = payload.get("fields", {})
        if isinstance(fields, dict):
            texts.extend(str(v) for v in fields.values())
        hints = payload.get("hints")
        if isinstance(hints, (list, tuple)):
            texts.extend(str(v) for v in hints)
        elif isinstance(hints, str):
            texts.append(hints)
        texts.append(payload.get("company_name", ""))
        best_label, scores = self.matcher.score(texts)
        final_label = best_label or "professional_services_consulting"
        description = self._fallback_description(payload, final_label)
        max_score = max(scores.values()) if scores else 0.0
        confidence = min(max_score / 5.0 + 0.2, 0.75) if max_score else 0.35
        return {
            "industry_slug": final_label,
            "confidence": confidence,
            "description": description,
            "provider": "pattern_fallback",
            "model": "regex",
            "cached": False,
            "used_fallback": True,
        }

    def _fallback_description(self, payload: Dict[str, object], label: str) -> str:
        parts: List[str] = []
        source = payload.get("source", "")
        if source:
            parts.append(f"Source={source}")
        fields = payload.get("fields", {})
        if isinstance(fields, dict):
            for key in ("sector", "industry", "description", "text_snippet"):
                value = fields.get(key)
                if isinstance(value, str) and value.strip():
                    trimmed = value.strip().replace("\n", " ")
                    parts.append(f"{key}: {trimmed}")
        fallback = " | ".join(parts)
        fallback = fallback[:400]
        if not fallback:
            fallback = f"Classified as {label} using fallback patterns."
        return fallback


class IndustryResolver:
    """Convenience wrapper to classify events consistently."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        map_path: str = "config/industry_map.yml",
        cache_dir: str | Path = "data/interim/industry_llm",
    ) -> None:
        self.classifier = IndustryLLMClassifier(
            model=model,
            provider=provider,
            map_path=map_path,
            cache_dir=cache_dir,
        )

    def classify_event(
        self,
        *,
        company_name: str,
        source: str,
        country: str = "",
        fields: Optional[Dict[str, object]] = None,
        hints: Optional[Sequence[str]] = None,
    ) -> ClassificationOutput:
        payload = {
            "company_name": company_name,
            "source": source,
            "country": country,
            "fields": fields or {},
            "hints": list(hints or []),
        }
        return self.classifier.classify(payload)

    @property
    def labels(self) -> Sequence[str]:
        return self.classifier.matcher.labels
