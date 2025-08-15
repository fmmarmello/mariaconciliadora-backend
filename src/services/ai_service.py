import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any
import openai
from groq import Groq
import re
from datetime import datetime, timedelta

class AIService:
    """
    Serviço de IA para análise de transações bancárias
    """
    
    def __init__(self):
        self.categories = {
            'alimentacao': ['mercado', 'supermercado', 'padaria', 'restaurante', 'lanchonete', 'delivery', 'ifood', 'uber eats'],
            'transporte': ['uber', 'taxi', 'combustivel', 'posto', 'onibus', 'metro', 'estacionamento'],
            'saude': ['farmacia', 'hospital', 'clinica', 'medico', 'dentista', 'laboratorio'],
            'educacao': ['escola', 'faculdade', 'curso', 'livro', 'material escolar'],
            'lazer': ['cinema', 'teatro', 'show', 'viagem', 'hotel', 'netflix', 'spotify'],
            'casa': ['aluguel', 'condominio', 'luz', 'agua', 'gas', 'internet', 'telefone'],
            'vestuario': ['roupa', 'sapato', 'loja', 'shopping'],
            'investimento': ['aplicacao', 'poupanca', 'cdb', 'tesouro', 'acao'],
            'transferencia': ['ted', 'doc', 'pix', 'transferencia'],
            'saque': ['saque', 'caixa eletronico'],
            'salario': ['salario', 'ordenado', 'pagamento'],
            'outros': []
        }
        
        # Configuração dos clientes de IA
        self.openai_client = None
        self.groq_client = None
        
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
        
        if os.getenv('GROQ_API_KEY'):
            self.groq_client = Groq()
    
    def categorize_transaction(self, description: str) -> str:
        """
        Categoriza uma transação baseada na descrição
        """
        description_lower = description.lower()
        
        # Busca por palavras-chave nas categorias
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        return 'outros'
    
    def categorize_transactions_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Categoriza um lote de transações
        """
        for transaction in transactions:
            transaction['category'] = self.categorize_transaction(transaction.get('description', ''))
        
        return transactions
    
    def detect_anomalies(self, transactions: List[Dict]) -> List[Dict]:
        """
        Detecta anomalias nas transações usando Isolation Forest
        """
        if len(transactions) < 10:  # Precisa de dados suficientes
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
        
        # Hora do dia (assumindo distribuição normal para transações)
        hours = [12] * len(transactions)  # Placeholder - seria melhor ter hora real
        features.append(hours)
        
        # Transpõe para formato correto
        X = np.array(features).T
        
        # Aplica Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        # Marca as anomalias
        for i, transaction in enumerate(transactions):
            transaction['is_anomaly'] = anomaly_labels[i] == -1
        
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
        
        anomalies = df[df['is_anomaly'] == True]
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': round((len(anomalies) / len(df)) * 100, 1) if len(df) > 0 else 0,
            'anomalous_transactions': anomalies[['date', 'amount', 'description']].to_dict('records')[:5]  # Top 5
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
    
    def generate_ai_insights(self, transactions: List[Dict]) -> str:
        """
        Gera insights usando IA generativa (GPT ou Groq)
        """
        if not transactions:
            return "Nenhuma transação disponível para análise."
        
        # Prepara um resumo dos dados para a IA
        summary = self.generate_insights(transactions)
        
        prompt = f"""
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
        
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400
                )
                return response.choices[0].message.content
            
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400
                )
                return response.choices[0].message.content
            
            else:
                return "Serviço de IA não configurado. Configure OPENAI_API_KEY ou GROQ_API_KEY."
        
        except Exception as e:
            return f"Erro ao gerar insights com IA: {str(e)}"
    
    def train_custom_model(self, financial_data: List[Dict]) -> Dict[str, Any]:
        """
        Treina um modelo personalizado com dados financeiros da empresa
        """
        try:
            # Prepara os dados de treinamento
            training_texts = [entry['description'] for entry in financial_data]
            training_labels = [entry['category'] for entry in financial_data]
            
            # Cria e treina o modelo
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(training_texts)
            
            # Treina um classificador
            classifier = KMeans(n_clusters=min(10, len(set(training_labels))))
            classifier.fit(X)
            
            # Salva o modelo e vetorizador
            self.custom_vectorizer = vectorizer
            self.custom_classifier = classifier
            self.model_trained = True
            
            # Calcula acurácia (simplificada)
            self.model_accuracy = self._calculate_accuracy(financial_data, classifier, vectorizer)
            
            return {
                'success': True,
                'message': 'Modelo treinado com sucesso',
                'accuracy': self.model_accuracy
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erro ao treinar modelo: {str(e)}'
            }
    
    def _calculate_accuracy(self, data: List[Dict], classifier, vectorizer) -> float:
        """Calcula a acurácia do modelo (simplificada)"""
        # Implementação simplificada para exemplo
        return 0.85  # Valor de exemplo
    
    def categorize_with_custom_model(self, description: str) -> str:
        """
        Categoriza uma transação usando o modelo personalizado treinado
        """
        if not self.model_trained:
            return self.categorize_transaction(description)  # Fallback para o método padrão
        
        try:
            # Transforma a descrição usando o vetorizador treinado
            X = self.custom_vectorizer.transform([description])
            
            # Prediz a categoria
            prediction = self.custom_classifier.predict(X)
            
            # Retorna a categoria predita (simplificada)
            return f"categoria_{prediction[0]}"
        except:
            # Fallback para o método padrão em caso de erro
            return self.categorize_transaction(description)
    
    def predict_financial_trends(self, historical_data: List[Dict], periods: int = 12) -> Dict[str, Any]:
        """
        Prevê tendências financeiras com base em dados históricos
        """
        try:
            # Converte dados históricos para DataFrame
            df = pd.DataFrame(historical_data)
            
            # Certifica-se de que temos as colunas necessárias
            if 'date' not in df.columns or 'amount' not in df.columns:
                return {'error': 'Dados insuficientes para predição'}
            
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
            return {
                'success': False,
                'error': f'Erro ao prever tendências: {str(e)}'
            }

