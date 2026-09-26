"""Text normalization and compact blocking fields for business records."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from unidecode import unidecode


WORD = re.compile(r"[a-z0-9]+")
DIGITS = re.compile(r"\d+")
NATIVE = re.compile(r"[ऀ-෿]")
SPLIT = re.compile(r"[\s,.;:()\[\]{}&+/\\#\"'_|-]+")
TRANSLIT_PATH = Path(os.environ.get("AMZN_TRANSLIT", "models/translit.json"))

LEGAL_WORDS = {
    "and", "company", "co", "corporation", "corp", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "llc", "llp", "plc", "the",
    # French legal forms and articles.
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "et", "de", "la", "le",
    "les", "du", "des", "d", "l",
    # Suffixes the generator adds to French names, like "Services" or "& Sons" in English.
    "cie", "compagnie", "fils", "freres", "frs", "associes", "groupe", "developpement",
    "pc", "pllc", "lp",
}


def merge_initials(tokens: list[str]) -> list[str]:
    """Join runs of single letters: 's a r l' -> 'sarl', 'l l c' -> 'llc' (dotted legal forms)."""
    result: list[str] = []
    run: list[str] = []
    for token in tokens + [""]:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= 2:
            result.append("".join(run))
        else:
            result.extend(run)
        run = []
        if token:
            result.append(token)
    return result
ADDRESS_WORDS = {
    "avenue", "ave", "boulevard", "blvd", "building", "bldg", "city",
    "colony", "district", "drive", "dr", "floor", "fl", "highway", "hwy",
    "lane", "ln", "near", "no", "north", "n", "opposite", "road", "rd",
    "saint", "sector", "south", "st", "state", "street", "township",
    "west", "w", "east", "e", "unit", "apartment", "apt", "block",
    "rue", "allee", "place", "route", "impasse", "chemin", "quai", "bis",
    "de", "la", "le", "les", "du", "des", "d", "l",
}
ADDRESS_REPLACEMENTS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "bldg": "building",
    "apt": "apartment", "marg": "road",
}
FRENCH_REPLACEMENTS = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "all": "allee", "pl": "place", "rte": "route", "imp": "impasse",
    "ch": "chemin", "chem": "chemin", "sq": "square", "fbg": "faubourg", "st": "saint",
    "ste": "sainte", "crs": "cours", "pass": "passage", "res": "residence",
}
# Region or state spellings collapse to one token so "Texas" equals "TX" and a
# French department equals its region.
REGION_PHRASES = {
    "france": {
        "hauts de france": "frhdf", "nord": "frhdf", "pas de calais": "frhdf",
        "nouvelle aquitaine": "frnaq", "gironde": "frnaq",
        "pays de la loire": "frpdl", "loire atlantique": "frpdl",
    },
    "us": {
        "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
        "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
        "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
        "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
        "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
        "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
        "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
        "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
        "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
        "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
        "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
        "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
    },
    "india": {
        "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
        "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
        "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
        "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
        "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
        "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
        "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
        "delhi": "dl", "chandigarh": "ch", "jammu and kashmir": "jk", "puducherry": "py",
        "pondicherry": "py",
    },
}


@lru_cache(maxsize=1)
def translit() -> dict[str, str]:
    if TRANSLIT_PATH.exists():
        return json.loads(TRANSLIT_PATH.read_text())
    return {}


def words(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if text.lower() in {"nan", "null", "none"}:
        return []
    if text.isascii():
        return WORD.findall(text.lower())
    mapping = translit()
    result: list[str] = []
    for token in SPLIT.split(text.lower()):
        if not token:
            continue
        if NATIVE.search(token):
            token = mapping.get(token, token)
        if not token.isascii():
            token = unidecode(token)
        result.extend(WORD.findall(token.lower()))
    return result


def replace_regions(tokens: list[str], country: str) -> list[str]:
    phrases = REGION_PHRASES.get(country)
    if not phrases:
        return tokens
    result: list[str] = []
    i = 0
    while i < len(tokens):
        for size in (4, 3, 2, 1):
            phrase = " ".join(tokens[i:i + size])
            if len(tokens) - i >= size and phrase in phrases:
                result.append(phrases[phrase])
                i += size
                break
        else:
            result.append(tokens[i])
            i += 1
    return result


def anchors(tokens: list[str], ignored: set[str], count: int = 2) -> tuple[str, ...]:
    candidates = {token for token in tokens if token not in ignored and len(token) >= 3}
    selected = sorted(candidates, key=lambda token: (-len(token), token))[:count]
    return tuple(selected + [""] * (count - len(selected)))


def normalize_record(row: dict[str, str]) -> tuple[str, ...]:
    """Return normalized fields in RECORD_COLUMNS order."""
    country_tokens = words(row.get("country"))
    country = " ".join(country_tokens)
    name_tokens = merge_initials(words(row.get("business_name")))
    replacements = FRENCH_REPLACEMENTS if country == "france" else ADDRESS_REPLACEMENTS
    address_tokens = [replacements.get(token, token) for token in words(row.get("business_address"))]
    address_tokens = replace_regions(address_tokens, country)

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
