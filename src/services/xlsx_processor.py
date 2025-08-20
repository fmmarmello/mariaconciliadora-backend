import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re
from src.services.duplicate_detection_service import DuplicateDetectionService
import hashlib

class XLSXProcessor:
    """
    Processador de arquivos XLSX para despesas e receitas empresariais
    """
    
    def __init__(self):
        self.supported_columns = {
            'date': ['data', 'date', 'dia'],
            'description': ['descricao', 'description', 'histórico', 'historico'],
            'amount': ['valor', 'amount', 'value'],
            'category': ['categoria', 'category'],
            'cost_center': ['centro de custo', 'cost center'],
            'department': ['departamento', 'department'],
            'project': ['projeto', 'project'],
            'transaction_type': ['tipo', 'type', 'transaction type'],
            'observations': ['observações', 'observations', 'obs'],
            'monthly_report_value': ['valor para relat mensal', 'valor para relatório mensal', 'monthly report value']
        }
        self.duplicate_service = DuplicateDetectionService()
    
    def parse_xlsx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processa um arquivo XLSX e retorna os dados estruturados
        """
        try:
            # Lê o arquivo XLSX
            df = pd.read_excel(file_path)
            
            # Mapeia os nomes das colunas para os nomes padrão
            column_mapping = self._get_column_mapping(df.columns)

            # Renomeia as colunas do DataFrame
            df.rename(columns=column_mapping, inplace=True)
            
            # Converte para lista de dicionários
            financial_data = []
            for _, row in df.iterrows():
                entry = {
                    'date': self._parse_date(row.get('date')),
                    'description': str(row.get('description', '')),
                    'amount': self._parse_amount(row.get('amount')),
                    'category': str(row.get('category', '')),
                    'cost_center': str(row.get('cost_center', '')),
                    'department': str(row.get('department', '')),
                    'project': str(row.get('project', '')),
                    'transaction_type': self._determine_transaction_type(row),
                    'observations': str(row.get('observations', '')),
                    'monthly_report_value': self._parse_amount(row.get('monthly_report_value'))
                }
                if 'saldo' in entry['description'].lower():
                    #ignora se for saldo, aqui faremos esse calculo
                    continue
                financial_data.append(entry)
            
            return financial_data
            
        except Exception as e:
            raise Exception(f"Erro ao processar arquivo XLSX: {str(e)}")

    def _get_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Cria um mapeamento de nomes de colunas para os nomes padrão"""
        mapping = {}
        normalized_columns = {self._normalize_column_name(col): col for col in columns}

        for standard_name, possible_names in self.supported_columns.items():
            for possible_name in possible_names:
                normalized_possible_name = self._normalize_column_name(possible_name)
                if normalized_possible_name in normalized_columns:
                    mapping[normalized_columns[normalized_possible_name]] = standard_name
                    break # Pega a primeira correspondência
        return mapping

    def _normalize_column_name(self, column_name: str) -> str:
        """Normaliza nomes de colunas para padrão consistente"""
        column_name = str(column_name).strip().lower()
        # Remove acentos e caracteres especiais
        column_name = re.sub(r'[^a-zA-Z0-9\s]', '', column_name)
        return column_name
    
    def _parse_date(self, date_value) -> datetime:
        """Parse de datas em vários formatos"""
        if pd.isna(date_value):
            return None
        try:
            return pd.to_datetime(date_value).date()
        except:
            return None
    
    def _parse_amount(self, amount_value) -> float:
        """Parse de valores monetários"""
        if pd.isna(amount_value):
            return 0.0
        try:
            return float(amount_value)
        except:
            return 0.0
    
    def _determine_transaction_type(self, row) -> str:
        """Determina se é despesa ou receita"""
        transaction_type = row.get('tipo', '').lower()
        #TODO adicionar IA para interpretar tipos
        if transaction_type in ['despesa', 'expense', 'débito', 'debit', 'retirada sócio',  'impostos / tributos', 'tarifas bancárias', 'juros / multa', 'seguro', 'emprestimo']:
            return 'expense'
        elif transaction_type in ['reembolso','receita', 'income', 'crédito', 'credit','credito']:
            return 'income'
        else:
            # Determina pelo valor (negativo = despesa, positivo = receita)
            amount = self._parse_amount(row.get('valor'))
            return 'expense' if amount < 0 else 'income'
    
    def detect_duplicates(self, entries: List[Dict]) -> List[int]:
        """
        Detecta possíveis entradas duplicadas
        Returns a list of indices of duplicate entries
        """
        duplicates = []
        seen = set()
        
        for i, entry in enumerate(entries):
            # Verifica se a entrada já existe no banco de dados
            is_duplicate = self.duplicate_service.check_financial_entry_duplicate(
                entry.get('date'),
                entry.get('amount'),
                entry.get('description')
            )
            
            if is_duplicate:
                duplicates.append(i)
                continue
            
            # Também verifica duplicatas dentro do próprio arquivo
            key = (
                entry.get('date'),
                entry.get('amount'),
                entry.get('description', '').lower().strip()
            )
            
            if key in seen:
                duplicates.append(i)
            else:
                seen.add(key)
        
        return duplicates