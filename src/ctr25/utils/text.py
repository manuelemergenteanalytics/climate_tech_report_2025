import re


def clean_text(s: str) -> str:
if not isinstance(s, str):
return ""
s = re.sub(r"\s+", " ", s).strip()
return s