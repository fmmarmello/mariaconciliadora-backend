"""
Shared financial constants and helpers.
"""

DEFAULT_COMPANY_FINANCIAL_CATEGORY = 'Nao categorizado'

CATEGORY_LABEL_OVERRIDES = {
    'folha_pagamento': 'Folha de Pagamento',
    'tarifas_bancarias': 'Tarifas Bancarias',
    'assinaturas_saas': 'Assinaturas SaaS',
    'contabilidade': 'Contabilidade',
    'fornecedores': 'Fornecedores',
    'impostos': 'Impostos',
    'logistica': 'Logistica',
    'marketing': 'Marketing',
    'manutencao': 'Manutencao',
    'equipamentos': 'Equipamentos',
    'juridico': 'Juridico',
    'seguros': 'Seguros',
    'ti': 'TI',
    'folha': 'Folha',
    'folha de pagamento': 'Folha de Pagamento'
}


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


def get_friendly_category_label(raw_value):
    """
    Provide a human-friendly label for financial categories while keeping a safe fallback.
    """
    normalized = normalize_company_financial_category(raw_value)
    if normalized == DEFAULT_COMPANY_FINANCIAL_CATEGORY:
        return normalized

    key = normalized.lower().replace('-', '_').replace(' ', '_')
    if key in CATEGORY_LABEL_OVERRIDES:
        return CATEGORY_LABEL_OVERRIDES[key]

    friendly = normalized.replace('_', ' ').replace('-', ' ').strip()
    if not friendly:
        return DEFAULT_COMPANY_FINANCIAL_CATEGORY

    return friendly.title()
