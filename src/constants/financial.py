"""
Shared financial constants and helpers.
"""

DEFAULT_COMPANY_FINANCIAL_CATEGORY = 'Nao categorizado'


def normalize_company_financial_category(raw_value):
    """
    Ensure a business-friendly category label is always available.

    Args:
        raw_value: Category value supplied by upstream sources.

    Returns:
        str: Trimmed category or the default placeholder when empty.
    """
    if raw_value is None:
        return DEFAULT_COMPANY_FINANCIAL_CATEGORY

    try:
        normalized = str(raw_value).strip()
    except Exception:
        normalized = ''

    if not normalized:
        return DEFAULT_COMPANY_FINANCIAL_CATEGORY

    lowered = normalized.lower()
    if lowered in {'nan', 'none', 'null', 'n/a', 'na'}:
        return DEFAULT_COMPANY_FINANCIAL_CATEGORY

    return normalized
