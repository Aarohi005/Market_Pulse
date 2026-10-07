"""
enums.py — structured enumerations for article classification.

Using Python enums (not string constants) gives us:
- IDE autocompletion
- Safe comparisons
- Serialisable .value for SQLite storage
"""
from __future__ import annotations
from enum import Enum


class SourceType(str, Enum):
    NEWSAPI = "NEWSAPI"
    RSS = "RSS"
    REUTERS = "REUTERS"
    CNBC = "CNBC"
    YAHOO = "YAHOO"
    MARKETWATCH = "MARKETWATCH"
    BLOOMBERG = "BLOOMBERG"
    UNKNOWN = "UNKNOWN"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    MACRO = "MACRO"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    COMMODITIES = "COMMODITIES"
    FIXED_INCOME = "FIXED_INCOME"
    UNKNOWN = "UNKNOWN"


class Region(str, Enum):
    US = "US"
    EU = "EU"
    APAC = "APAC"
    GLOBAL = "GLOBAL"
    UNKNOWN = "UNKNOWN"


class Importance(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SentimentCategory(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Topic(str, Enum):
    EARNINGS = "EARNINGS"
    FED_POLICY = "FED_POLICY"
    INFLATION = "INFLATION"
    RECESSION = "RECESSION"
    AI = "AI"
    TECH = "TECH"
    BANKING = "BANKING"
    ENERGY = "ENERGY"
    GEOPOLITICS = "GEOPOLITICS"
    CRYPTO = "CRYPTO"
    HOUSING = "HOUSING"
    LABOR = "LABOR"
    TRADE = "TRADE"
    OTHER = "OTHER"
