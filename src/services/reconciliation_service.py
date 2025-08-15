import pandas as pd
from datetime import datetime, timedelta
from src.models.transaction import Transaction, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from typing import List, Dict, Any

class ReconciliationService:
    """
    Serviço para reconciliação entre transações bancárias e entradas financeiras da empresa
    """
    
    def __init__(self):
        pass
    
    def find_matches(self, bank_transactions: List[Transaction], company_entries: List[CompanyFinancial]) -> List[Dict[str, Any]]:
        """
        Encontra correspondências entre transações bancárias e entradas financeiras da empresa
        """
        matches = []
        
        for bank_transaction in bank_transactions:
            best_match = None
            best_score = 0.0
            
            for company_entry in company_entries:
                score = self._calculate_match_score(bank_transaction, company_entry)
                
                if score > best_score and score >= 0.7:  # Limiar mínimo para considerar match
                    best_score = score
                    best_match = company_entry
            
            if best_match:
                matches.append({
                    'bank_transaction': bank_transaction,
                    'company_entry': best_match,
                    'match_score': best_score
                })
        
        return matches
    
    def _calculate_match_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """
        Calcula o score de correspondência entre uma transação bancária e uma entrada financeira
        """
        score = 0.0
        
        # Verifica se os valores são iguais ou muito próximos
        if abs(abs(bank_transaction.amount) - abs(company_entry.amount)) < 0.01:
            score += 0.4
        elif abs(abs(bank_transaction.amount) - abs(company_entry.amount)) < 1.0:
            score += 0.2
        
        # Verifica se as datas são próximas (até 3 dias de diferença)
        if bank_transaction.date and company_entry.date:
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff == 0:
                score += 0.3
            elif date_diff <= 3:
                score += 0.15
        
        # Verifica similaridade nas descrições
        description_similarity = self._calculate_description_similarity(
            bank_transaction.description, 
            company_entry.description
        )
        score += description_similarity * 0.3
        
        return min(score, 1.0)  # Limita o score a 1.0
    
    def _calculate_description_similarity(self, bank_description: str, company_description: str) -> float:
        """
        Calcula a similaridade entre duas descrições usando uma abordagem simples
        """
        # Converte para minúsculas e remove espaços extras
        bank_desc = bank_description.lower().strip()
        company_desc = company_description.lower().strip()
        
        # Verifica se uma descrição está contida na outra
        if bank_desc in company_desc or company_desc in bank_desc:
            return 0.8
        
        # Verifica palavras em comum
        bank_words = set(bank_desc.split())
        company_words = set(company_desc.split())
        
        if len(bank_words) > 0 and len(company_words) > 0:
            common_words = bank_words.intersection(company_words)
            similarity = len(common_words) / max(len(bank_words), len(company_words))
            return similarity * 0.6
        
        return 0.0
    
    def create_reconciliation_records(self, matches: List[Dict[str, Any]]) -> List[ReconciliationRecord]:
        """
        Cria registros de reconciliação no banco de dados
        """
        records = []
        
        for match in matches:
            record = ReconciliationRecord(
                bank_transaction_id=match['bank_transaction'].id,
                company_entry_id=match['company_entry'].id,
                match_score=match['match_score'],
                status='pending'
            )
            db.session.add(record)
            records.append(record)
        
        db.session.commit()
        return records
    
    def get_pending_reconciliations(self) -> List[ReconciliationRecord]:
        """
        Obtém todos os registros de reconciliação pendentes
        """
        return ReconciliationRecord.query.filter_by(status='pending').all()
    
    def confirm_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Confirma uma reconciliação
        """
        record = ReconciliationRecord.query.get(reconciliation_id)
        if record:
            record.status = 'confirmed'
            db.session.commit()
            return True
        return False
    
    def reject_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Rejeita uma reconciliação
        """
        record = ReconciliationRecord.query.get(reconciliation_id)
        if record:
            record.status = 'rejected'
            db.session.commit()
            return True
        return False
    
    def get_reconciliation_report(self) -> Dict[str, Any]:
        """
        Gera um relatório de reconciliação
        """
        total_records = ReconciliationRecord.query.count()
        confirmed_records = ReconciliationRecord.query.filter_by(status='confirmed').count()
        pending_records = ReconciliationRecord.query.filter_by(status='pending').count()
        rejected_records = ReconciliationRecord.query.filter_by(status='rejected').count()
        
        # Calcula valores reconciliados
        confirmed_matches = ReconciliationRecord.query.filter_by(status='confirmed').all()
        total_reconciled_value = sum(
            abs(record.bank_transaction.amount) for record in confirmed_matches 
            if record.bank_transaction
        )
        
        return {
            'summary': {
                'total_records': total_records,
                'confirmed': confirmed_records,
                'pending': pending_records,
                'rejected': rejected_records,
                'reconciliation_rate': confirmed_records / total_records if total_records > 0 else 0
            },
            'financials': {
                'total_reconciled_value': round(total_reconciled_value, 2)
            }
        }