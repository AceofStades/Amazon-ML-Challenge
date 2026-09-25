"""Text normalization and compact blocking fields for business records."""

from __future__ import annotations

import re

from unidecode import unidecode


WORD = re.compile(r"[a-z0-9]+")
DIGITS = re.compile(r"\d+")

LEGAL_WORDS = {
    "and", "company", "co", "corporation", "corp", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "llc", "llp", "plc", "the",
}
ADDRESS_WORDS = {
    "avenue", "ave", "boulevard", "blvd", "building", "bldg", "city",
    "colony", "district", "drive", "dr", "floor", "fl", "highway", "hwy",
    "lane", "ln", "near", "no", "north", "n", "opposite", "road", "rd",
    "saint", "sector", "south", "st", "state", "street", "township",
    "west", "w", "east", "e", "unit", "apartment", "apt", "block",
}
ADDRESS_REPLACEMENTS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "bldg": "building",
    "apt": "apartment", "marg": "road",
}


def words(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if text.lower() in {"nan", "null", "none"}:
        return []
    if not text.isascii():
        text = unidecode(text)
    return WORD.findall(text.lower())


def anchors(tokens: list[str], ignored: set[str], count: int = 2) -> tuple[str, ...]:
    candidates = {token for token in tokens if token not in ignored and len(token) >= 3}
    selected = sorted(candidates, key=lambda token: (-len(token), token))[:count]
    return tuple(selected + [""] * (count - len(selected)))


def normalize_record(row: dict[str, str]) -> tuple[str, ...]:
    """Return normalized fields in RECORD_COLUMNS order."""
    name_tokens = words(row.get("business_name"))
    address_tokens = [ADDRESS_REPLACEMENTS.get(token, token) for token in words(row.get("business_address"))]
    country_tokens = words(row.get("country"))
    country = " ".join(country_tokens)

    core_name_tokens = [token for token in name_tokens if token not in LEGAL_WORDS]
    if not core_name_tokens:
        core_name_tokens = name_tokens
    core_name = " ".join(sorted(core_name_tokens))
    name1, name2 = anchors(core_name_tokens, set())

    address1, address2 = anchors(address_tokens, ADDRESS_WORDS)
    numeric = DIGITS.findall(" ".join(address_tokens))
    house = str(int(numeric[0])) if numeric else ""

    return (
        row["entity_id"], country, " ".join(name_tokens), " ".join(address_tokens),
        core_name, " ".join(sorted(set(address_tokens))), house,
        name1, name2, address1, address2,
    )


RECORD_COLUMNS = (
    "entity_id", "country", "name", "address", "name_core", "address_core",
    "house", "name_anchor1", "name_anchor2", "address_anchor1", "address_anchor2",
)
