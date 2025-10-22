import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional
import openai
from groq import Groq
import re
import time
from datetime import datetime, timedelta
import unicodedata
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    AIServiceError, AIServiceUnavailableError, AIServiceTimeoutError,
    AIServiceQuotaExceededError, InsufficientDataError, ValidationError
)
from src.utils.error_handler import handle_service_errors, with_timeout, recovery_manager

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

class AIService:
    """
    Serviço de IA para análise de transações bancárias
    """
    
    def __init__(self):
        # Categorias base + expansões para rotina de PMEs brasileiras.
        # Mantém categorias existentes e adiciona novas sem remover nenhuma.
        self.categories = {
            # Existentes
            'alimentacao': ['mercado', 'supermercado', 'padaria', 'restaurante', 'lanchonete', 'delivery', 'ifood', 'uber eats', 'rappi'],
            'transporte': ['uber', '99', '99pop', 'taxi', 'combustivel', 'posto', 'onibus', 'metro', 'estacionamento', 'pedagio', 'pedágio'],
            'saude': ['farmacia', 'hospital', 'clinica', 'clínica', 'medico', 'médico', 'dentista', 'laboratorio', 'laboratório', 'exame'],
            'educacao': ['escola', 'faculdade', 'curso', 'livro', 'material escolar', 'ead', 'treinamento'],
            'lazer': ['cinema', 'teatro', 'show', 'viagem', 'hotel', 'netflix', 'spotify', 'lazer'],
            'casa': ['aluguel', 'condominio', 'luz', 'energia', 'agua', 'água', 'gas', 'gás', 'internet', 'telefone', 'claro', 'vivo', 'tim', 'oi'],
            'vestuario': ['roupa', 'sapato', 'loja', 'shopping', 'uniforme'],
            'investimento': ['aplicacao', 'aplicação', 'poupanca', 'poupança', 'cdb', 'tesouro', 'acao', 'ação'],
            'transferencia': ['ted', 'doc', 'pix', 'transferencia', 'transferência', 'boleto'],
            'saque': ['saque', 'caixa eletronico', 'caixa eletrônico'],
            'salario': ['salario', 'salário', 'ordenado', 'pagamento'],

            # Novas categorias orientadas a PMEs
            'impostos': ['darf', 'das', 'simples', 'mei', 'iss', 'icms', 'irpj', 'csll', 'pis', 'cofins', 'gps', 'inss', 'sefaz', 'prefeitura', 'taxa', 'alvara', 'alvará', 'guia'],
            'contabilidade': ['contabilidade', 'contador', 'escritorio contabil', 'escritório contábil', 'honorarios contabeis', 'honorários contábeis', 'balanco', 'balanço'],
            'folha_pagamento': ['folha', 'pro labore', 'prolabore', 'pró-labore', 'fgts', 'vale refeicao', 'vale refeição', 'vr', 'vale alimentacao', 'vale alimentação', 'va', 'vale transporte', 'vt', '13o', '13º', 'ferias', 'férias'],
            'fornecedores': ['fornecedor', 'fornecedores', 'compra de materiais', 'insumos', 'materiais', 'mercadoria', 'estoque', 'materia-prima', 'matéria-prima', 'compra', 'nfe', 'nf-e', 'nota fiscal de compra'],
            'logistica': ['frete', 'transportadora', 'correios', 'sedex', 'pac', 'jadlog', 'loggi', 'envio', 'entrega', 'motoboy', 'coleta', 'despacho'],
            'marketing': ['marketing', 'publicidade', 'anuncio', 'anúncio', 'ads', 'facebook ads', 'google ads', 'meta ads', 'instagram ads', 'linkedin ads', 'campanha', 'impulsionamento'],
            'tarifas_bancarias': ['tarifa', 'cesta de servicos', 'cesta de serviços', 'manutencao de conta', 'manutenção de conta', 'taxa bancaria', 'taxa bancária', 'pix tarifa', 'ted tarifa', 'doc tarifa', 'boleto tarifa'],
            'assinaturas_saas': ['assinatura', 'subscription', 'mensalidade', 'plano', 'licenca', 'licença', 'saas', 'software', 'ferramenta', 'google workspace', 'microsoft 365', 'office 365', 'notion', 'slack', 'zoom', 'github', 'gitlab'],
            'juridico': ['juridico', 'jurídico', 'advogado', 'escritorio advocacia', 'escritório advocacia', 'honorarios', 'honorários'],
            'manutencao': ['manutencao', 'manutenção', 'reparo', 'conserto', 'assistencia tecnica', 'assistência técnica', 'suporte tecnico', 'suporte técnico', 'calibracao', 'calibração'],
            'ti': ['aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'cloudflare', 'hospedagem', 'dominio', 'domínio', 'registro br', 'digitalocean', 'linode', 'vultr', 'heroku', 'railway', 'servidor', 'dns'],
            'seguros': ['seguro', 'apolice', 'apólice', 'premio', 'prêmio', 'seguradora', 'porto seguro', 'sulamerica', 'sulamérica', 'bradesco seguros'],
            'equipamentos': ['equipamento', 'impressora', 'computador', 'laptop', 'notebook', 'teclado', 'mouse', 'monitor', 'hardware', 'periferico', 'periférico', 'celular', 'smartphone', 'tablet'],
            'limpeza': ['limpeza', 'higiene', 'desinfetante', 'material de limpeza'],
            'escritorio': ['papelaria', 'cartucho', 'toner', 'caneta', 'grampeador', 'pastas', 'envelope', 'impressoes', 'impressões', 'xerox', 'copias', 'cópias'],

            'outros': []
        }
        
        # Configuração dos clientes de IA
        self.openai_client = None
        self.groq_client = None
        
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
            logger.info("OpenAI client initialized")
        
        if os.getenv('GROQ_API_KEY'):
            self.groq_client = Groq()
            logger.info("Groq client initialized")
        
        if not self.openai_client and not self.groq_client:
            logger.warning("No AI service configured. Set OPENAI_API_KEY or GROQ_API_KEY for AI features")
        
        # AI service configuration
        self.default_timeout = int(os.getenv('AI_SERVICE_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('AI_SERVICE_MAX_RETRIES', '3'))
        self.rate_limit_delay = float(os.getenv('AI_SERVICE_RATE_LIMIT_DELAY', '1.0'))
        
        # Model persistence configuration
        self.model_dir = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'))
        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception:
            pass
        self._load_persisted_model()

    def _model_paths(self):
        vectorizer_path = os.path.join(self.model_dir, 'custom_vectorizer.joblib')
        model_path = os.path.join(self.model_dir, 'custom_classifier.joblib')
        meta_path = os.path.join(self.model_dir, 'custom_model_meta.json')
        return vectorizer_path, model_path, meta_path

    def _load_persisted_model(self):
        try:
            vectorizer_path, model_path, meta_path = self._model_paths()
            if os.path.exists(vectorizer_path) and os.path.exists(model_path):
                self.custom_vectorizer = joblib.load(vectorizer_path)
                self.custom_classifier = joblib.load(model_path)
                self.model_trained = True
                # Load meta if available
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self.model_meta = json.load(f)
                logger.info("Custom AI model loaded from disk")
            else:
                self.model_trained = False
        except Exception as e:
            logger.warning(f"Failed to load persisted model, falling back. Error: {e}")
            self.model_trained = False

    def _save_persisted_model(self, vectorizer, classifier, meta: Dict[str, Any]):
        try:
            vectorizer_path, model_path, meta_path = self._model_paths()
            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(classifier, model_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False)
            logger.info(f"Custom AI model saved to {self.model_dir}")
        except Exception as e:
            logger.warning(f"Failed to persist model to disk: {e}")
    
    def _normalize_text(self, text: str) -> str:
        try:
            return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').lower()
        except Exception:
            return str(text).lower()

    def categorize_transaction(self, description: str) -> str:
        """
        Categoriza uma transação baseada na descrição
        """
        description_lower = self._normalize_text(description)
        
        # Busca por palavras-chave nas categorias
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if self._normalize_text(keyword) in description_lower:
                    return category
        
        return 'outros'

    def _rule_based_override(self, description: str) -> Optional[str]:
        """Return a high-confidence rule-based category for override decisions.
        Uses specific, non-generic keywords to avoid false positives (e.g., 'pagamento').
        """
        text = self._normalize_text(description)
        checks = [
            ('impostos', ['darf','das','simples','irpj','csll','pis','cofins','iss','icms','sefaz','prefeitura']),
            ('tarifas_bancarias', ['tarifa','cesta de servicos','manutencao de conta','manutencao','cesta de servicos']),
            ('assinaturas_saas', ['assinatura','subscription','google workspace','microsoft 365','office 365','notion','slack','zoom','github','gitlab']),
            ('ti', ['aws','azure','gcp','google cloud','cloudflare','dominio','hospedagem','registro br','dns']),
            ('fornecedores', ['fornecedor','insumos','materia-prima','mercadoria','estoque','nfe','nf-e']),
            ('logistica', ['frete','transportadora','correios','sedex','pac','jadlog','loggi','motoboy']),
            ('contabilidade', ['contabilidade','contador','honorarios contabeis']),
            ('juridico', ['juridico','advogado','escritorio advocacia']),
            ('manutencao', ['manutencao','reparo','conserto','assistencia tecnica','calibracao']),
            ('seguros', ['seguro','apolice']),
            ('escritorio', ['papelaria','toner','cartucho','caneta']),
            ('equipamentos', ['impressora','computador','notebook','monitor','teclado','mouse']),
            ('folha_pagamento', ['prolabore','fgts','13o','13']),
        ]
        for cat, kws in checks:
            for kw in kws:
                if kw in text:
                    return cat
        return None
    
    @handle_service_errors('ai_service')
    @with_timeout(120)  # 2 minute timeout for batch processing
    def categorize_transactions_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Categoriza um lote de transações
        """
        if not transactions:
            raise InsufficientDataError('transaction categorization', 1, 0)
        
        logger.info(f"Starting batch categorization for {len(transactions)} transactions")
        
        try:
            categorized_count = 0
            failed_count = 0
            
            for i, transaction in enumerate(transactions):
                try:
                    description = transaction.get('description', '')
                    if not description or description.strip() == '':
                        transaction['category'] = 'outros'
                        logger.debug(f"Transaction {i+1}: Empty description, assigned 'outros' category")
                    else:
                        transaction['category'] = self.categorize_transaction(description)
                        categorized_count += 1
                        
                    # Add small delay to avoid overwhelming the system
                    if i > 0 and i % 100 == 0:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to categorize transaction {i+1}: {str(e)}")
                    transaction['category'] = 'outros'  # Fallback category
                    failed_count += 1
            
            logger.info(f"Batch categorization completed. Success: {categorized_count}, Failed: {failed_count}")
            audit_logger.log_ai_operation('batch_categorization', len(transactions), True)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Critical error in batch categorization: {str(e)}")
            audit_logger.log_ai_operation('batch_categorization', len(transactions), False, error=str(e))
            raise AIServiceError(f"Batch categorization failed: {str(e)}")
    
    @handle_service_errors('ai_service')
    @with_timeout(60)  # 1 minute timeout for anomaly detection
    def detect_anomalies(self, transactions: List[Dict]) -> List[Dict]:
        """
        Detecta anomalias nas transações usando Isolation Forest
        """
        if not transactions:
            raise InsufficientDataError('anomaly detection', 1, 0)
        
        logger.info(f"Starting anomaly detection for {len(transactions)} transactions")
        
        if len(transactions) < 10:  # Precisa de dados suficientes
            logger.warning("Insufficient data for anomaly detection (minimum 10 transactions required)")
            # Mark all as non-anomalous
            for transaction in transactions:
                transaction['is_anomaly'] = False
            return transactions
        
        # Prepara os dados para análise
        df = pd.DataFrame(transactions)
        
        # Features para detecção de anomalias
        features = []
        
        # Valor absoluto da transação
        amounts = [abs(t.get('amount', 0)) for t in transactions]
        features.append(amounts)
        
        # Dia da semana (se tiver data)
        weekdays = []
        for t in transactions:
            if t.get('date'):
                if isinstance(t['date'], str):
                    date_obj = datetime.fromisoformat(t['date'])
                else:
                    date_obj = t['date']
                weekdays.append(date_obj.weekday())
            else:
                weekdays.append(0)
        features.append(weekdays)
        
        # Hora do dia baseada em 'timestamp' (quando existir) ou 'date'
        hours = []
        for t in transactions:
            hour_val = 12  # fallback neutro
            try:
                # Preferir timestamp com hora completa
                ts = t.get('timestamp')
                if ts is not None:
                    if isinstance(ts, str):
                        ts_obj = datetime.fromisoformat(ts)
                        hour_val = getattr(ts_obj, 'hour', 12)
                    elif hasattr(ts, 'hour'):
                        hour_val = ts.hour
                else:
                    # Fallback para 'date'
                    d = t.get('date')
                    if isinstance(d, str):
                        d_obj = datetime.fromisoformat(d)
                        hour_val = getattr(d_obj, 'hour', 0)
                    elif hasattr(d, 'hour'):
                        hour_val = d.hour
                    else:
                        hour_val = 0
            except Exception:
                hour_val = 12
            hours.append(hour_val)
        features.append(hours)
        
        # Transpõe para formato correto
        X = np.array(features).T
        
        # Aplica Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        # Marca as anomalias
        anomaly_count = 0
        for i, transaction in enumerate(transactions):
            is_anomaly = anomaly_labels[i] == -1
            transaction['is_anomaly'] = is_anomaly
            if is_anomaly:
                anomaly_count += 1
        
        logger.info(f"Anomaly detection completed. Found {anomaly_count} anomalies out of {len(transactions)} transactions")
        audit_logger.log_ai_operation('anomaly_detection', len(transactions), True)
        return transactions
    
    def generate_insights(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Gera insights sobre as transações
        """
        if not transactions:
            return {'error': 'Nenhuma transação para análise'}
        
        df = pd.DataFrame(transactions)
        
        insights = {
            'summary': self._generate_summary_insights(df),
            'categories': self._generate_category_insights(df),
            'patterns': self._generate_pattern_insights(df),
            'anomalies': self._generate_anomaly_insights(df),
            'recommendations': []
        }
        
        # Adiciona recomendações baseadas nos insights
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def _generate_summary_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights resumidos
        """
        total_transactions = len(df)
        total_credits = df[df['amount'] > 0]['amount'].sum() if len(df[df['amount'] > 0]) > 0 else 0
        total_debits = abs(df[df['amount'] < 0]['amount'].sum()) if len(df[df['amount'] < 0]) > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'total_credits': round(total_credits, 2),
            'total_debits': round(total_debits, 2),
            'net_flow': round(total_credits - total_debits, 2),
            'avg_transaction_value': round(df['amount'].abs().mean(), 2) if total_transactions > 0 else 0,
            'largest_expense': round(df[df['amount'] < 0]['amount'].min(), 2) if len(df[df['amount'] < 0]) > 0 else 0,
            'largest_income': round(df[df['amount'] > 0]['amount'].max(), 2) if len(df[df['amount'] > 0]) > 0 else 0
        }
    
    def _generate_category_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights por categoria
        """
        if 'category' not in df.columns:
            return {}
        
        category_summary = {}
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            category_expenses = category_df[category_df['amount'] < 0]['amount'].sum()
            
            category_summary[category] = {
                'total_transactions': len(category_df),
                'total_spent': round(abs(category_expenses), 2),
                'avg_transaction': round(category_df['amount'].abs().mean(), 2),
                'percentage_of_expenses': 0  # Será calculado depois
            }
        
        # Calcula percentuais
        total_expenses = sum([cat['total_spent'] for cat in category_summary.values()])
        if total_expenses > 0:
            for category in category_summary:
                category_summary[category]['percentage_of_expenses'] = round(
                    (category_summary[category]['total_spent'] / total_expenses) * 100, 1
                )
        
        return category_summary
    
    def _generate_pattern_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights sobre padrões temporais
        """
        patterns = {}
        
        if 'date' in df.columns and len(df) > 0:
            # Converte datas se necessário
            dates = []
            for date_val in df['date']:
                if isinstance(date_val, str):
                    try:
                        dates.append(datetime.fromisoformat(date_val))
                    except:
                        dates.append(datetime.now())
                else:
                    dates.append(date_val)
            
            df['parsed_date'] = dates
            df['weekday'] = df['parsed_date'].apply(lambda x: x.weekday())
            df['day'] = df['parsed_date'].apply(lambda x: x.day)
            
            # Padrões por dia da semana
            weekday_spending = df[df['amount'] < 0].groupby('weekday')['amount'].sum().abs()
            weekdays = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            
            patterns['weekday_spending'] = {
                weekdays[i]: round(weekday_spending.get(i, 0), 2) 
                for i in range(7)
            }
            
            # Dia do mês com mais gastos
            day_spending = df[df['amount'] < 0].groupby('day')['amount'].sum().abs()
            if len(day_spending) > 0:
                patterns['highest_spending_day'] = int(day_spending.idxmax())
                patterns['highest_spending_amount'] = round(day_spending.max(), 2)
        
        return patterns
    
    def _generate_anomaly_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights sobre anomalias
        """
        if 'is_anomaly' not in df.columns:
            return {}
        
        anomalies = df[df['is_anomaly'] == True].copy()

        # Compute context stats for simple explanations
        try:
            abs_amounts = df['amount'].abs().astype(float)
            amt_mean = float(abs_amounts.mean()) if len(abs_amounts) else 0.0
            amt_std = float(abs_amounts.std(ddof=0)) if len(abs_amounts) else 0.0
        except Exception:
            amt_mean, amt_std = 0.0, 0.0

        # Weekend ratio
        weekend_ratio = 0.0
        try:
            def to_weekday(row):
                d = row.get('date') if isinstance(row, dict) else row
                if isinstance(d, str):
                    try:
                        return datetime.fromisoformat(d).weekday()
                    except Exception:
                        return None
                return None
            weekdays_series = df.apply(lambda r: to_weekday({'date': r.get('date') if hasattr(r, 'get') else r['date']}), axis=1) if hasattr(df, 'apply') else []
            valid_weekdays = [w for w in weekdays_series if isinstance(w, int)] if len(df) else []
            weekend_count = sum(1 for w in valid_weekdays if w >= 5)
            weekend_ratio = (weekend_count / len(valid_weekdays)) if valid_weekdays else 0.0
        except Exception:
            weekend_ratio = 0.0

        def derive_hour(row) -> int:
            # Prefer timestamp, then try date
            ts = row.get('timestamp') if isinstance(row, dict) else None
            d = row.get('date') if isinstance(row, dict) else None
            try:
                if ts:
                    if isinstance(ts, str):
                        return datetime.fromisoformat(ts).hour
                    if hasattr(ts, 'hour'):
                        return int(ts.hour)
                if d:
                    if isinstance(d, str):
                        return getattr(datetime.fromisoformat(d), 'hour', 0)
                    if hasattr(d, 'hour'):
                        return int(d.hour)
            except Exception:
                return 12
            return 0

        # Build reasons for each anomaly
        records = []
        for _, row in anomalies.iterrows():
            reasons = []
            try:
                val = float(abs(row['amount'])) if 'amount' in row else 0.0
            except Exception:
                val = 0.0

            # Amount-based reason
            if amt_std and (val - amt_mean) > 2 * amt_std:
                reasons.append('Valor muito acima do padrão histórico')
            elif amt_std and (val - amt_mean) > 1.5 * amt_std:
                reasons.append('Valor acima do normal')

            # Time-based reason
            hour = derive_hour(row if isinstance(row, dict) else row.to_dict())
            if hour < 6 or hour > 22:
                reasons.append('Horário incomum (madrugada/noite)')

            # Weekend-based reason
            try:
                date_str = None
                if isinstance(row.get('date'), str):
                    date_str = row.get('date')
                elif hasattr(row.get('date'), 'isoformat'):
                    date_str = row.get('date').isoformat()
                if date_str:
                    wd = datetime.fromisoformat(date_str).weekday()
                    if wd >= 5 and weekend_ratio < 0.2:
                        reasons.append('Transação em fim de semana pouco comum')
            except Exception:
                pass

            if not reasons:
                reasons.append('Padrão incomum detectado pelo modelo')

            rec = {
                'date': row.get('date').isoformat() if hasattr(row.get('date'), 'isoformat') else row.get('date'),
                'amount': float(row.get('amount')) if row.get('amount') is not None else None,
                'description': row.get('description'),
                'anomaly_reason': ' • '.join(reasons)
            }
            # Include timestamp if present
            if 'timestamp' in df.columns:
                rec['timestamp'] = row.get('timestamp')
            records.append(rec)

        return {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': round((len(anomalies) / len(df)) * 100, 1) if len(df) > 0 else 0,
            'anomalous_transactions': records[:5]
        }
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """
        Gera recomendações baseadas nos insights
        """
        recommendations = []
        
        # Recomendações baseadas em categorias
        if 'categories' in insights:
            categories = insights['categories']
            
            # Categoria com maior gasto
            if categories:
                max_category = max(categories.items(), key=lambda x: x[1]['total_spent'])
                if max_category[1]['total_spent'] > 0:
                    recommendations.append(
                        f"Sua maior categoria de gastos é '{max_category[0]}' com R$ {max_category[1]['total_spent']:.2f}. "
                        f"Considere revisar esses gastos para possíveis economias."
                    )
        
        # Recomendações baseadas em anomalias
        if 'anomalies' in insights:
            anomaly_count = insights['anomalies'].get('total_anomalies', 0)
            if anomaly_count > 0:
                recommendations.append(
                    f"Foram detectadas {anomaly_count} transações incomuns. "
                    f"Revise essas transações para verificar se são legítimas."
                )
        
        # Recomendações baseadas no fluxo de caixa
        if 'summary' in insights:
            net_flow = insights['summary'].get('net_flow', 0)
            if net_flow < 0:
                recommendations.append(
                    f"Seu saldo líquido está negativo em R$ {abs(net_flow):.2f}. "
                    f"Considere reduzir gastos ou aumentar receitas."
                )
        
        return recommendations
    
    @handle_service_errors('ai_service')
    @with_timeout(45)  # 45 second timeout for AI insights
    def generate_ai_insights(self, transactions: List[Dict]) -> str:
        """
        Gera insights usando IA generativa (GPT ou Groq)
        """
        if not transactions:
            raise InsufficientDataError('AI insights generation', 1, 0)
        
        # Check if AI service is available
        if not self.openai_client and not self.groq_client:
            raise AIServiceUnavailableError('No AI service configured')
        
        try:
            # Prepara um resumo dos dados para a IA
            summary = self.generate_insights(transactions)
            
            prompt = self._create_insights_prompt(summary)
            
            logger.info("Generating AI insights using external AI service")
            
            # Try with retry mechanism
            result = self._generate_insights_with_retry(prompt, len(transactions))
            
            return result
            
        except Exception as e:
            if isinstance(e, (AIServiceError, InsufficientDataError)):
                raise
            
            logger.error(f"Unexpected error generating AI insights: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('insights_generation', len(transactions), False, error=str(e))
            raise AIServiceError(f"AI insights generation failed: {str(e)}")
    
    def _create_insights_prompt(self, summary: Dict[str, Any]) -> str:
        """Create a structured prompt for AI insights generation."""
        return f"""
        Analise os seguintes dados financeiros e forneça insights valiosos em português:
        
        Resumo das transações:
        - Total de transações: {summary['summary']['total_transactions']}
        - Total de receitas: R$ {summary['summary']['total_credits']:.2f}
        - Total de gastos: R$ {summary['summary']['total_debits']:.2f}
        - Saldo líquido: R$ {summary['summary']['net_flow']:.2f}
        
        Gastos por categoria:
        {summary['categories']}
        
        Forneça uma análise concisa e actionable sobre:
        1. Principais padrões de gastos
        2. Oportunidades de economia
        3. Alertas importantes
        4. Recomendações personalizadas
        
        Mantenha a resposta em até 300 palavras e seja prático.
        """
    
    def _generate_insights_with_retry(self, prompt: str, transaction_count: int) -> str:
        """Generate insights with retry mechanism and fallback."""
        def try_openai():
            if not self.openai_client:
                raise AIServiceUnavailableError('OpenAI')
            
            logger.debug("Using OpenAI GPT-4o-mini for insights generation")
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    timeout=self.default_timeout
                )
                result = response.choices[0].message.content
                audit_logger.log_ai_operation('insights_generation', transaction_count, True, 'gpt-4o-mini')
                return result
                
            except openai.RateLimitError:
                raise AIServiceQuotaExceededError('OpenAI')
            except openai.APITimeoutError:
                raise AIServiceTimeoutError('OpenAI', self.default_timeout)
            except openai.APIConnectionError:
                raise AIServiceUnavailableError('OpenAI')
        
        def try_groq():
            if not self.groq_client:
                raise AIServiceUnavailableError('Groq')
            
            logger.debug("Using Groq Llama3 for insights generation")
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400
                )
                result = response.choices[0].message.content
                audit_logger.log_ai_operation('insights_generation', transaction_count, True, 'llama3-8b-8192')
                return result
                
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    raise AIServiceQuotaExceededError('Groq')
                elif 'timeout' in str(e).lower():
                    raise AIServiceTimeoutError('Groq', self.default_timeout)
                else:
                    raise AIServiceUnavailableError('Groq')
        
        def fallback_insights():
            logger.warning("All AI services failed, generating fallback insights")
            return self._generate_fallback_insights(transaction_count)
        
        # Try primary service with retry
        for attempt in range(self.max_retries):
            try:
                if self.openai_client:
                    return try_openai()
                elif self.groq_client:
                    return try_groq()
            except (AIServiceQuotaExceededError, AIServiceTimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    logger.warning(f"AI service error, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"AI service failed after {self.max_retries} attempts")
                    break
            except AIServiceUnavailableError:
                # Try alternative service
                try:
                    if self.groq_client and attempt == 0:
                        return try_groq()
                    elif self.openai_client and attempt == 0:
                        return try_openai()
                except Exception:
                    pass
                break
        
        # Use fallback if all services fail
        return fallback_insights()
    
    def _generate_fallback_insights(self, transaction_count: int) -> str:
        """Generate basic insights when AI services are unavailable."""
        return f"""
        Análise básica dos dados financeiros:
        
        📊 Resumo: {transaction_count} transações processadas
        
        ⚠️ Serviço de IA temporariamente indisponível
        
        Recomendações gerais:
        • Revise regularmente suas transações
        • Categorize gastos para melhor controle
        • Monitore transações incomuns
        • Mantenha registros organizados
        
        Para análises mais detalhadas, tente novamente em alguns minutos.
        """
    
    @handle_service_errors('ai_service')
    @with_timeout(300)  # 5 minute timeout for model training
    def train_custom_model(self, financial_data: List[Dict]) -> Dict[str, Any]:
        """
        Treina um modelo personalizado com dados financeiros da empresa
        """
        if not financial_data:
            raise InsufficientDataError('model training', 10, 0)
        
        if len(financial_data) < 10:
            raise InsufficientDataError('model training', 10, len(financial_data))
        
        try:
            logger.info(f"Starting custom model training with {len(financial_data)} data points")
            
            # Validate training data
            valid_data = self._validate_training_data(financial_data)
            
            if len(valid_data) < 5:
                raise InsufficientDataError('model training', 5, len(valid_data))
            
            # Prepara os dados de treinamento
            training_texts = [entry['description'] for entry in valid_data]
            training_labels = [entry.get('category') or 'outros' for entry in valid_data]

            unique_categories = len(set(training_labels))
            logger.info(f"Training data prepared. Valid entries: {len(valid_data)}, Unique categories: {unique_categories}")

            # Vetorização de texto
            vectorizer = TfidfVectorizer(
                max_features=min(5000, max(500, len(valid_data) * 5)),
                min_df=1,
                max_df=0.95,
                stop_words=None
            )
            X = vectorizer.fit_transform(training_texts)

            # Escolhe abordagem: supervisionada (preferível) ou KMeans fallback
            use_supervised = unique_categories >= 2 and len(valid_data) >= 10
            metrics: Dict[str, Any] = {}
            start_time = time.time()
            if use_supervised:
                classifier = LinearSVC()
                # Split para validação
                test_size = 0.2 if len(valid_data) >= 30 else 0.15
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, training_labels, test_size=test_size, random_state=42, stratify=training_labels if len(set(training_labels)) > 1 else None
                    )
                except ValueError:
                    # Fallback sem estratificação
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, training_labels, test_size=test_size, random_state=42
                    )
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                # Relatório por classe (pode falhar com poucas amostras)
                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    category_accuracy = {cls: float(metrics_dict.get('precision', 0.0)) for cls, metrics_dict in report.items() if cls not in ('accuracy', 'macro avg', 'weighted avg')}
                except Exception:
                    category_accuracy = {}
                metrics['validation_samples'] = len(y_test)
                metrics['category_accuracy'] = category_accuracy
            else:
                # Fallback: KMeans como clusterizador (não supervisionado)
                n_clusters = max(1, min(10, unique_categories if unique_categories > 0 else 1, max(1, len(valid_data) // 2)))
                classifier = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                classifier.fit(X)
                # Mapeia clusters para categorias mais frequentes para uma métrica aproximada
                try:
                    labels = classifier.labels_
                    cluster_to_label: Dict[int, str] = {}
                    correct = 0
                    total = len(training_labels)
                    for cluster_id in set(labels):
                        indices = [i for i, c in enumerate(labels) if c == cluster_id]
                        cats = [training_labels[i] for i in indices]
                        if cats:
                            majority = max(set(cats), key=cats.count)
                            cluster_to_label[cluster_id] = majority
                    for i, cluster_id in enumerate(labels):
                        mapped = cluster_to_label.get(cluster_id, 'outros')
                        if mapped == training_labels[i]:
                            correct += 1
                    acc = float(correct / total) if total else 0.0
                    # Per-category approximation
                    cat_correct: Dict[str, int] = {}
                    cat_total: Dict[str, int] = {}
                    for i, true_cat in enumerate(training_labels):
                        cat_total[true_cat] = cat_total.get(true_cat, 0) + 1
                        cluster_id = labels[i]
                        mapped = cluster_to_label.get(cluster_id, 'outros')
                        if mapped == true_cat:
                            cat_correct[true_cat] = cat_correct.get(true_cat, 0) + 1
                    category_accuracy = {cat: float(cat_correct.get(cat, 0) / total_c) for cat, total_c in cat_total.items() if total_c > 0}
                except Exception:
                    acc = 0.0
                    category_accuracy = {}
                metrics['category_accuracy'] = category_accuracy

            elapsed_ms = int((time.time() - start_time) * 1000)
            metrics['training_time_ms'] = elapsed_ms

            # Salva o modelo e vetorizador
            self.custom_vectorizer = vectorizer
            self.custom_classifier = classifier
            self.model_trained = True
            self.model_accuracy = acc

            # Persistência em disco
            meta = {
                'accuracy': acc,
                'training_data_count': len(valid_data),
                'categories_count': unique_categories,
                'trained_at': datetime.utcnow().isoformat(),
                'algorithm': 'LinearSVC' if use_supervised else 'KMeans',
                'metrics': metrics,
                'labels': sorted(list(set(training_labels))),
            }
            self._save_persisted_model(vectorizer, classifier, meta)

            logger.info(f"Custom model training completed successfully. Accuracy: {acc:.2f}")
            audit_logger.log_ai_operation('model_training', len(financial_data), True)

            return {
                'success': True,
                'message': 'Modelo treinado com sucesso',
                'accuracy': acc,
                'training_data_count': len(valid_data),
                'categories_count': unique_categories,
                'metrics': metrics,
            }
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            
            logger.error(f"Error training custom model: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('model_training', len(financial_data), False, error=str(e))
            raise AIServiceError(f'Model training failed: {str(e)}')
    
    def _validate_training_data(self, financial_data: List[Dict]) -> List[Dict]:
        """Validate and clean training data."""
        valid_data = []
        
        for entry in financial_data:
            # Check required fields
            if not entry.get('description') or not entry.get('description').strip():
                continue
            
            # Clean description
            description = str(entry['description']).strip()
            if len(description) < 3:  # Too short to be meaningful
                continue
            
            # Add to valid data
            valid_entry = entry.copy()
            valid_entry['description'] = description
            valid_data.append(valid_entry)
        
        return valid_data
    
    def _calculate_accuracy(self, data: List[Dict], classifier, vectorizer) -> float:
        """Mantido por compatibilidade: calcula acurácia simples em holdout quando possível."""
        try:
            texts = [d['description'] for d in data]
            labels = [d.get('category') or 'outros' for d in data]
            X = vectorizer.transform(texts) if hasattr(vectorizer, 'vocabulary_') else vectorizer.fit_transform(texts)
            if hasattr(classifier, 'predict'):
                # No training done here; assume already trained
                preds = classifier.predict(X)
                return float(accuracy_score(labels, preds))
        except Exception:
            pass
        return 0.0
    
    @handle_service_errors('ai_service')
    def categorize_with_custom_model(self, description: str) -> str:
        """
        Categoriza uma transação usando o modelo personalizado treinado
        """
        if not description or not description.strip():
            return 'outros'
        
        if not hasattr(self, 'model_trained') or not self.model_trained:
            # Tenta carregar do disco
            self._load_persisted_model()
            if not getattr(self, 'model_trained', False):
                logger.debug("Custom model not trained, using fallback categorization")
                return self.categorize_transaction(description)
        
        try:
            # Validate model components
            if not hasattr(self, 'custom_vectorizer') or not hasattr(self, 'custom_classifier'):
                logger.warning("Custom model components missing, using fallback")
                return self.categorize_transaction(description)
            
            # Clean and prepare description
            clean_description = str(description).strip()
            
            # Transforma a descrição usando o vetorizador treinado
            X = self.custom_vectorizer.transform([clean_description])
            
            # Prediz a categoria
            prediction = self.custom_classifier.predict(X)

            # Se for KMeans, mapear cluster -> categoria; caso contrário, usar rótulo diretamente
            if isinstance(self.custom_classifier, KMeans):
                category = self._map_cluster_to_category(int(prediction[0]), clean_description)
            else:
                category = str(prediction[0])
                # Optional override: prefer high-confidence rule-based category not seen in training labels
                try:
                    rule_based = self._rule_based_override(clean_description)
                    labels = set(self.model_meta.get('labels', [])) if hasattr(self, 'model_meta') and isinstance(self.model_meta, dict) else set()
                    if rule_based and rule_based not in labels:
                        category = rule_based
                except Exception:
                    pass
            
            logger.debug(f"Custom model categorized '{clean_description}' as '{category}'")
            return category
            
        except Exception as e:
            logger.warning(f"Error using custom model for categorization: {str(e)}")
            # Fallback para o método padrão em caso de erro
            return self.categorize_transaction(description)
    
    def _map_cluster_to_category(self, cluster_id: int, description: str) -> str:
        """Map cluster ID to meaningful category name."""
        # Prefer rule-based categorization first for more stable results
        rule_based_category = self.categorize_transaction(description)
        if rule_based_category != 'outros':
            return rule_based_category

        # Fallback: deterministic mapping cycling through a richer set of SME categories
        ordered = [
            'impostos', 'fornecedores', 'contabilidade', 'marketing', 'logistica', 'ti',
            'assinaturas_saas', 'tarifas_bancarias', 'manutencao', 'juridico', 'seguros',
            'equipamentos', 'escritorio', 'limpeza', 'folha_pagamento',
            # categorias base
            'alimentacao', 'transporte', 'casa', 'saude', 'lazer', 'vestuario',
            'educacao', 'investimento', 'transferencia', 'saque'
        ]
        if not ordered:
            return 'outros'
        idx = abs(int(cluster_id)) % len(ordered)
        return ordered[idx]
    
    @handle_service_errors('ai_service')
    @with_timeout(60)  # 1 minute timeout for predictions
    def predict_financial_trends(self, historical_data: List[Dict], periods: int = 12) -> Dict[str, Any]:
        """
        Prevê tendências financeiras com base em dados históricos
        """
        if not historical_data:
            raise InsufficientDataError('financial trend prediction', 30, 0)
        
        if len(historical_data) < 30:
            raise InsufficientDataError('financial trend prediction', 30, len(historical_data))
        
        if periods <= 0 or periods > 24:
            raise ValidationError("Periods must be between 1 and 24")
        
        try:
            # Converte dados históricos para DataFrame
            df = pd.DataFrame(historical_data)
            
            # Validate required columns
            required_columns = ['date', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            
            # Clean and validate data
            df = self._clean_historical_data(df)
            
            if len(df) < 10:
                raise InsufficientDataError('financial trend prediction', 10, len(df))
            
            # Converte datas
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Prepara dados para predição (simplificada)
            # Numa implementação real, isso usaria modelos de séries temporais
            
            # Calcula tendências básicas
            total_income = df[df['amount'] > 0]['amount'].sum() if len(df[df['amount'] > 0]) > 0 else 0
            total_expenses = abs(df[df['amount'] < 0]['amount'].sum()) if len(df[df['amount'] < 0]) > 0 else 0
            net_flow = total_income - total_expenses
            
            # Calcula médias mensais
            df['month'] = df['date'].dt.to_period('M')
            monthly_data = df.groupby('month')['amount'].sum().reset_index()
            avg_monthly_income = monthly_data[monthly_data['amount'] > 0]['amount'].mean() if len(monthly_data[monthly_data['amount'] > 0]) > 0 else 0
            avg_monthly_expenses = abs(monthly_data[monthly_data['amount'] < 0]['amount'].mean()) if len(monthly_data[monthly_data['amount'] < 0]) > 0 else 0
            
            # Previsões para próximos períodos (simplificadas)
            predictions = []
            current_date = df['date'].max() if len(df) > 0 else pd.Timestamp.now()
            
            for i in range(1, periods + 1):
                future_date = current_date + pd.DateOffset(months=i)
                predicted_income = avg_monthly_income if not pd.isna(avg_monthly_income) else total_income / len(monthly_data) if len(monthly_data) > 0 else 0
                predicted_expenses = avg_monthly_expenses if not pd.isna(avg_monthly_expenses) else total_expenses / len(monthly_data) if len(monthly_data) > 0 else 0
                predicted_net_flow = predicted_income - predicted_expenses
                
                predictions.append({
                    'date': future_date.strftime('%Y-%m'),
                    'predicted_income': round(predicted_income, 2),
                    'predicted_expenses': round(predicted_expenses, 2),
                    'predicted_net_flow': round(predicted_net_flow, 2)
                })
            
            return {
                'success': True,
                'data': {
                    'historical_summary': {
                        'total_income': round(total_income, 2),
                        'total_expenses': round(total_expenses, 2),
                        'net_flow': round(net_flow, 2),
                        'period_months': len(monthly_data)
                    },
                    'predictions': predictions
                }
            }
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            
            logger.error(f"Error predicting financial trends: {str(e)}", exc_info=True)
            raise AIServiceError(f'Financial trend prediction failed: {str(e)}')
    
    def _clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate historical data for predictions."""
        # Remove rows with missing critical data
        df = df.dropna(subset=['date', 'amount'])
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convert amounts to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Remove extreme outliers (amounts beyond reasonable range)
        amount_q99 = df['amount'].quantile(0.99)
        amount_q01 = df['amount'].quantile(0.01)
        df = df[(df['amount'] >= amount_q01) & (df['amount'] <= amount_q99)]
        
        # Sort by date
        df = df.sort_values('date')
        
        return df

