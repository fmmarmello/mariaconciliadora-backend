import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from datetime import datetime

from src.main import app, db  # ensures app context and DB init
from src.models.transaction import Transaction
from src.services.ai_service import AIService


def generate_preview(bank=None, start_date=None, end_date=None, category=None, only_outros=False, limit=300, out_path=None):
    with app.app_context():
        query = Transaction.query
        if bank:
            query = query.filter(Transaction.bank_name == bank)
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_dt.date())
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_dt.date())
        if category:
            query = query.filter(Transaction.category == category)
        if only_outros:
            query = query.filter((Transaction.category == 'outros') | (Transaction.category.is_(None)))
        query = query.order_by(Transaction.date.desc())
        txs = query.limit(max(1, min(int(limit), 5000))).all()
        if not txs:
            print('No transactions found for given filters')
            return None

        ai = AIService()
        use_custom = getattr(ai, 'model_trained', False)
        if not use_custom:
            try:
                ai._load_persisted_model()
                use_custom = getattr(ai, 'model_trained', False)
            except Exception:
                use_custom = False

        total = 0
        changed = 0
        unchanged = 0
        transitions = {}
        examples = []
        new_categories = set()

        def inc(d, k):
            d[k] = d.get(k, 0) + 1

        for t in txs:
            total += 1
            desc = t.description or ''
            if not desc.strip():
                predicted = 'outros'
            else:
                try:
                    if use_custom:
                        predicted = ai.categorize_with_custom_model(desc)
                    else:
                        predicted = ai.categorize_transaction(desc)
                except Exception:
                    predicted = 'outros'
            old = t.category if t.category is not None else 'NULL'
            new = predicted or 'outros'
            if new != (t.category or 'NULL'):
                changed += 1
                if len(examples) < 50:
                    examples.append({
                        'id': t.id,
                        'date': t.date.isoformat() if t.date else None,
                        'amount': t.amount,
                        'bank': t.bank_name,
                        'description': t.description,
                        'old': t.category,
                        'new': new,
                    })
            else:
                unchanged += 1
            inc(transitions, f"{old} -> {new}")
            if new and new not in (None, '', 'outros', old):
                new_categories.add(new)

        generated_at = datetime.utcnow().isoformat()
        model_info = 'custom_trained_model' if use_custom else 'rule_based_fallback'

        lines = []
        lines.append('# Previa de Reprocessamento de Categorias')
        lines.append('')
        lines.append(f'Gerado em: {generated_at} (UTC)')
        lines.append(f'Modelo: {model_info}')
        lines.append('')
        lines.append('## Filtros Utilizados')
        lines.append(f'- Banco: {bank or "todos"}')
        lines.append(f'- Periodo: {(start_date or "inicio")} -> {(end_date or "fim")}')
        lines.append(f'- Categoria atual: {category or "todas"}; only_outros={only_outros}')
        lines.append(f'- Limite analisado: {len(txs)}')
        lines.append('')
        lines.append('## Sumario')
        lines.append(f'- Total analisadas: {total}')
        lines.append(f'- Alteradas: {changed}')
        lines.append(f'- Mantidas: {unchanged}')
        if new_categories:
            lines.append(f"- Novas categorias sugeridas: {', '.join(sorted(list(new_categories)))}")
        lines.append('')
        lines.append('## Transicoes (antiga -> nova)')
        lines.append('')
        lines.append('| De | Para | Qtde |')
        lines.append('|---|---:|---:|')
        for key, count in sorted(transitions.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            try:
                old_v, new_v = key.split(' -> ', 1)
            except ValueError:
                old_v, new_v = key, ''
            lines.append(f'| {old_v} | {new_v} | {count} |')
        lines.append('')
        lines.append('## Exemplos de Alteracoes (ate 50)')
        lines.append('')
        lines.append('| ID | Data | Banco | Valor | Categoria Antiga | Categoria Nova | Descricao |')
        lines.append('|---:|:-----|:------|------:|:------------------|:---------------|:----------|')
        for ex in examples:
            date_str = ex['date'] or ''
            bank_str = ex['bank'] or ''
            amount_str = f"{ex['amount']:.2f}" if isinstance(ex['amount'], (int, float)) else str(ex['amount'])
            old_str = ex['old'] if ex['old'] is not None else 'NULL'
            new_str = ex['new']
            desc_str = (ex['description'] or '').replace('|', '¦')
            lines.append(f"| {ex['id']} | {date_str} | {bank_str} | {amount_str} | {old_str} | {new_str} | {desc_str} |")

        report_md = '\n'.join(lines)

        if out_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            docs_dir = os.path.join(repo_root, 'docs')
            os.makedirs(docs_dir, exist_ok=True)
            out_path = os.path.join(docs_dir, 'REPROCESSAMENTO_CATEGORIAS_PREVIEW.md')

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report_md)

        print(f'Report saved to: {out_path}')
        return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate reprocess categories preview markdown')
    parser.add_argument('--bank', default=None)
    parser.add_argument('--start-date', dest='start_date', default=None)
    parser.add_argument('--end-date', dest='end_date', default=None)
    parser.add_argument('--category', default=None)
    parser.add_argument('--only-outros', dest='only_outros', action='store_true')
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--out', dest='out_path', default=None)

    args = parser.parse_args()
    generate_preview(
        bank=args.bank,
        start_date=args.start_date,
        end_date=args.end_date,
        category=args.category,
        only_outros=args.only_outros,
        limit=args.limit,
        out_path=args.out_path,
    )


