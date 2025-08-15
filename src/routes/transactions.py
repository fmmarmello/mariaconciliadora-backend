from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime, timedelta
from src.models.transaction import Transaction, UploadHistory, ReconciliationRecord, db
from src.models.company_financial import CompanyFinancial
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService

transactions_bp = Blueprint('transactions', __name__)

# Configurações de upload
ALLOWED_EXTENSIONS = {'ofx', 'qfx'}
ALLOWED_XLSX_EXTENSIONS = {'xlsx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_xlsx_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_XLSX_EXTENSIONS

@transactions_bp.route('/upload-ofx', methods=['POST'])
def upload_ofx():
    """
    Endpoint para upload e processamento de arquivos OFX
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não permitido. Use apenas .ofx ou .qfx'}), 400
        
        # Salva o arquivo temporariamente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Processa o arquivo OFX
        processor = OFXProcessor()
        ai_service = AIService()
        
        try:
            # Parse do OFX
            ofx_data = processor.parse_ofx_file(temp_path)
            
            # Valida as transações
            valid_transactions = processor.validate_transactions(ofx_data['transactions'])
            
            # Detecta duplicatas
            duplicates = processor.detect_duplicates(valid_transactions)
            
            # Aplica IA para categorização e detecção de anomalias
            categorized_transactions = ai_service.categorize_transactions_batch(valid_transactions)
            analyzed_transactions = ai_service.detect_anomalies(categorized_transactions)
            
            # Salva as transações no banco de dados
            saved_count = 0
            for transaction_data in analyzed_transactions:
                # Verifica se a transação já existe (evita duplicatas)
                existing = Transaction.query.filter_by(
                    account_id=ofx_data['account_info'].get('account_id', ''),
                    date=transaction_data['date'],
                    amount=transaction_data['amount'],
                    description=transaction_data['description']
                ).first()
                
                if not existing:
                    transaction = Transaction(
                        bank_name=ofx_data['bank_name'],
                        account_id=ofx_data['account_info'].get('account_id', ''),
                        transaction_id=transaction_data.get('transaction_id', ''),
                        date=transaction_data['date'],
                        amount=transaction_data['amount'],
                        description=transaction_data['description'],
                        transaction_type=transaction_data['transaction_type'],
                        balance=transaction_data.get('balance'),
                        category=transaction_data.get('category'),
                        is_anomaly=transaction_data.get('is_anomaly', False)
                    )
                    db.session.add(transaction)
                    saved_count += 1
            
            # Salva o histórico de upload
            upload_record = UploadHistory(
                filename=filename,
                bank_name=ofx_data['bank_name'],
                transactions_count=saved_count,
                status='success'
            )
            db.session.add(upload_record)
            
            db.session.commit()
            
            # Remove o arquivo temporário
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return jsonify({
                'success': True,
                'message': f'Arquivo processado com sucesso! {saved_count} transações importadas.',
                'data': {
                    'bank_name': ofx_data['bank_name'],
                    'account_info': ofx_data['account_info'],
                    'transactions_imported': saved_count,
                    'duplicates_found': len(duplicates),
                    'summary': ofx_data['summary']
                }
            })
            
        except Exception as e:
            # Salva o erro no histórico
            upload_record = UploadHistory(
                filename=filename,
                bank_name='unknown',
                transactions_count=0,
                status='error',
                error_message=str(e)
            )
            db.session.add(upload_record)
            db.session.commit()
            
            # Remove o arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
                os.rmdir(temp_dir)
            
            return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@transactions_bp.route('/transactions', methods=['GET'])
def get_transactions():
    """
    Endpoint para listar transações com filtros
    """
    try:
        # Parâmetros de filtro
        bank_name = request.args.get('bank')
        category = request.args.get('category')
        transaction_type = request.args.get('type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Constrói a query
        query = Transaction.query
        
        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)
        
        if category:
            query = query.filter(Transaction.category == category)
        
        if transaction_type:
            query = query.filter(Transaction.transaction_type == transaction_type)
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date())
        
        # Ordena por data (mais recente primeiro)
        query = query.order_by(Transaction.date.desc())
        
        # Aplica paginação
        transactions = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        return jsonify({
            'success': True,
            'data': {
                'transactions': [t.to_dict() for t in transactions],
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar transações: {str(e)}'}), 500

@transactions_bp.route('/insights', methods=['GET'])
def get_insights():
    """
    Endpoint para obter insights das transações
    """
    try:
        # Parâmetros de filtro (mesmo que get_transactions)
        bank_name = request.args.get('bank')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Constrói a query
        query = Transaction.query
        
        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date())
        
        transactions = query.all()
        
        if not transactions:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Nenhuma transação encontrada para o período selecionado'
                }
            })
        
        # Converte para formato de dicionário
        transactions_data = [t.to_dict() for t in transactions]
        
        # Gera insights
        ai_service = AIService()
        insights = ai_service.generate_insights(transactions_data)
        
        return jsonify({
            'success': True,
            'data': insights
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar insights: {str(e)}'}), 500

@transactions_bp.route('/ai-insights', methods=['GET'])
def get_ai_insights():
    """
    Endpoint para obter insights gerados por IA
    """
    try:
        # Parâmetros de filtro
        bank_name = request.args.get('bank')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Constrói a query (mesmo que get_insights)
        query = Transaction.query
        
        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date())
        
        transactions = query.all()
        
        if not transactions:
            return jsonify({
                'success': True,
                'data': {
                    'ai_insights': 'Nenhuma transação encontrada para análise.'
                }
            })
        
        # Converte para formato de dicionário
        transactions_data = [t.to_dict() for t in transactions]
        
        # Gera insights com IA
        ai_service = AIService()
        ai_insights = ai_service.generate_ai_insights(transactions_data)
        
        return jsonify({
            'success': True,
            'data': {
                'ai_insights': ai_insights
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar insights com IA: {str(e)}'}), 500

@transactions_bp.route('/summary', methods=['GET'])
def get_summary():
    """
    Endpoint para obter resumo geral das transações
    """
    try:
        # Estatísticas gerais
        total_transactions = Transaction.query.count()
        total_credits = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.amount > 0).scalar() or 0
        total_debits = abs(db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.amount < 0).scalar() or 0)
        
        # Transações por banco
        banks = db.session.query(
            Transaction.bank_name,
            db.func.count(Transaction.id).label('count')
        ).group_by(Transaction.bank_name).all()
        
        # Transações por categoria
        categories = db.session.query(
            Transaction.category,
            db.func.count(Transaction.id).label('count'),
            db.func.sum(Transaction.amount).label('total')
        ).group_by(Transaction.category).all()
        
        # Anomalias
        anomalies_count = Transaction.query.filter(Transaction.is_anomaly == True).count()
        
        # Últimas transações
        recent_transactions = Transaction.query.order_by(Transaction.date.desc()).limit(5).all()
        
        return jsonify({
            'success': True,
            'data': {
                'overview': {
                    'total_transactions': total_transactions,
                    'total_credits': round(total_credits, 2),
                    'total_debits': round(total_debits, 2),
                    'net_flow': round(total_credits - total_debits, 2),
                    'anomalies_count': anomalies_count
                },
                'banks': [{'name': bank.bank_name, 'count': bank.count} for bank in banks],
                'categories': [
                    {
                        'name': cat.category,
                        'count': cat.count,
                        'total': round(cat.total or 0, 2)
                    } for cat in categories
                ],
                'recent_transactions': [t.to_dict() for t in recent_transactions]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar resumo: {str(e)}'}), 500

@transactions_bp.route('/upload-history', methods=['GET'])
def get_upload_history():
    """
    Endpoint para obter histórico de uploads
    """
    try:
        uploads = UploadHistory.query.order_by(UploadHistory.upload_date.desc()).limit(20).all()
        
        return jsonify({
            'success': True,
            'data': {
                'uploads': [upload.to_dict() for upload in uploads]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar histórico: {str(e)}'}), 500

@transactions_bp.route('/upload-xlsx', methods=['POST'])
def upload_xlsx():
    """
    Endpoint para upload e processamento de arquivos XLSX
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_xlsx_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não permitido. Use apenas .xlsx'}), 400
        
        # Salva o arquivo temporariamente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Processa o arquivo XLSX
        processor = XLSXProcessor()
        financial_data = processor.parse_xlsx_file(temp_path)
        
        # Salva os dados no banco de dados
        saved_count = 0
        for entry_data in financial_data:
            financial_entry = CompanyFinancial(
                date=entry_data['date'],
                description=entry_data['description'],
                amount=entry_data['amount'],
                category=entry_data['category'],
                cost_center=entry_data['cost_center'],
                department=entry_data['department'],
                project=entry_data['project'],
                transaction_type=entry_data['transaction_type']
            )
            db.session.add(financial_entry)
            saved_count += 1
        
        db.session.commit()
        
        # Remove o arquivo temporário
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'success': True,
            'message': f'Arquivo processado com sucesso! {saved_count} entradas importadas.',
            'data': {
                'entries_imported': saved_count
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500

@transactions_bp.route('/company-financial', methods=['GET'])
def get_company_financial():
    """
    Endpoint para listar entradas financeiras da empresa com filtros
    """
    try:
        # Parâmetros de filtro
        transaction_type = request.args.get('type')  # expense, income, or all
        category = request.args.get('category')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Constrói a query
        query = CompanyFinancial.query
        
        if transaction_type and transaction_type != 'all':
            query = query.filter(CompanyFinancial.transaction_type == transaction_type)
        
        if category:
            query = query.filter(CompanyFinancial.category == category)
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Ordena por data (mais recente primeiro)
        query = query.order_by(CompanyFinancial.date.desc())
        
        # Aplica paginação
        entries = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        return jsonify({
            'success': True,
            'data': {
                'entries': [e.to_dict() for e in entries],
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar entradas financeiras: {str(e)}'}), 500

@transactions_bp.route('/company-financial/summary', methods=['GET'])
def get_company_financial_summary():
    """
    Endpoint para obter resumo geral das entradas financeiras
    """
    try:
        # Parâmetros de filtro
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Constrói a query base
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Aplica os filtros para estatísticas
        filtered_entries = query.all()
        
        # Estatísticas gerais
        total_entries = len(filtered_entries)
        total_income = sum(entry.amount for entry in filtered_entries if entry.transaction_type == 'income')
        total_expenses = sum(entry.amount for entry in filtered_entries if entry.transaction_type == 'expense')
        
        # Entradas por categoria
        category_stats = {}
        for entry in filtered_entries:
            if entry.category not in category_stats:
                category_stats[entry.category] = {'count': 0, 'total': 0}
            category_stats[entry.category]['count'] += 1
            category_stats[entry.category]['total'] += entry.amount
        
        # Últimas entradas
        recent_entries = sorted(filtered_entries, key=lambda x: x.date, reverse=True)[:5]
        
        return jsonify({
            'success': True,
            'data': {
                'overview': {
                    'total_entries': total_entries,
                    'total_income': round(total_income, 2),
                    'total_expenses': round(total_expenses, 2),
                    'net_flow': round(total_income - total_expenses, 2)
                },
                'categories': [
                    {
                        'name': category,
                        'count': stats['count'],
                        'total': round(stats['total'], 2)
                    } for category, stats in category_stats.items()
                ],
                'recent_entries': [e.to_dict() for e in recent_entries]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar resumo: {str(e)}'}), 500

@transactions_bp.route('/ai/train', methods=['POST'])
def train_ai_model():
    """
    Endpoint para treinar o modelo de IA com dados financeiros da empresa
    """
    try:
        # Parâmetros de filtro para selecionar dados de treinamento
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        category = request.args.get('category')
        
        # Constrói a query para dados de treinamento
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        if category:
            query = query.filter(CompanyFinancial.category == category)
        
        # Obtém os dados de treinamento
        training_data = query.all()
        
        if not training_data:
            return jsonify({
                'success': False,
                'error': 'Nenhum dado encontrado para treinamento'
            }), 400
        
        # Converte para formato de dicionário
        training_data_dicts = [entry.to_dict() for entry in training_data]
        
        # Treina o modelo
        ai_service = AIService()
        result = ai_service.train_custom_model(training_data_dicts)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Modelo treinado com sucesso',
                'accuracy': result['accuracy']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Erro ao treinar modelo: {str(e)}'}), 500

@transactions_bp.route('/ai/categorize-financial', methods=['POST'])
def categorize_financial_with_ai():
    """
    Endpoint para categorizar entradas financeiras usando o modelo treinado
    """
    try:
        # Obtém os dados para categorização
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Descrição não fornecida'}), 400
        
        description = data['description']
        
        # Categoriza usando o modelo treinado
        ai_service = AIService()
        category = ai_service.categorize_with_custom_model(description)
        
        return jsonify({
            'success': True,
            'data': {
                'category': category
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao categorizar: {str(e)}'}), 500

@transactions_bp.route('/ai/predictions', methods=['GET'])
def get_financial_predictions():
    """
    Endpoint para obter previsões financeiras baseadas em dados históricos
    """
    try:
        # Parâmetros para dados históricos
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        periods = request.args.get('periods', 12, type=int)
        
        # Constrói a query para dados históricos
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Obtém os dados históricos
        historical_data = query.all()
        
        if not historical_data:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Nenhum dado encontrado para predição'
                }
            })
        
        # Converte para formato de dicionário
        historical_data_dicts = [entry.to_dict() for entry in historical_data]
        
        # Gera previsões
        ai_service = AIService()
        predictions = ai_service.predict_financial_trends(historical_data_dicts, periods)
        
        if predictions['success']:
            return jsonify({
                'success': True,
                'data': predictions['data']
            })
        else:
            return jsonify({
                'success': False,
                'error': predictions['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar previsões: {str(e)}'}), 500

@transactions_bp.route('/reconciliation', methods=['POST'])
def start_reconciliation():
    """
    Endpoint para iniciar o processo de reconciliação entre transações bancárias e entradas financeiras
    """
    try:
        # Parâmetros para filtrar dados
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Constrói queries para dados bancários e financeiros
        bank_query = Transaction.query
        company_query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            bank_query = bank_query.filter(Transaction.date >= start_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            bank_query = bank_query.filter(Transaction.date <= end_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Obtém os dados
        bank_transactions = bank_query.all()
        company_entries = company_query.all()
        
        # Realiza a reconciliação
        reconciliation_service = ReconciliationService()
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        records = reconciliation_service.create_reconciliation_records(matches)
        
        return jsonify({
            'success': True,
            'message': f'Reconciliação iniciada com sucesso. {len(records)} correspondências encontradas.',
            'data': {
                'matches_count': len(records)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao iniciar reconciliação: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/pending', methods=['GET'])
def get_pending_reconciliations():
    """
    Endpoint para obter registros de reconciliação pendentes
    """
    try:
        reconciliation_service = ReconciliationService()
        pending_records = reconciliation_service.get_pending_reconciliations()
        
        return jsonify({
            'success': True,
            'data': {
                'records': [record.to_dict() for record in pending_records]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar reconciliações pendentes: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/confirm', methods=['POST'])
def confirm_reconciliation(reconciliation_id):
    """
    Endpoint para confirmar uma reconciliação
    """
    try:
        reconciliation_service = ReconciliationService()
        success = reconciliation_service.confirm_reconciliation(reconciliation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Reconciliação confirmada com sucesso'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Reconciliação não encontrada'
            }), 404
            
    except Exception as e:
        return jsonify({'error': f'Erro ao confirmar reconciliação: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/reject', methods=['POST'])
def reject_reconciliation(reconciliation_id):
    """
    Endpoint para rejeitar uma reconciliação
    """
    try:
        reconciliation_service = ReconciliationService()
        success = reconciliation_service.reject_reconciliation(reconciliation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Reconciliação rejeitada com sucesso'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Reconciliação não encontrada'
            }), 404
            
    except Exception as e:
        return jsonify({'error': f'Erro ao rejeitar reconciliação: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/report', methods=['GET'])
def get_reconciliation_report():
    """
    Endpoint para obter relatório de reconciliação
    """
    try:
        reconciliation_service = ReconciliationService()
        report = reconciliation_service.get_reconciliation_report()
        
        return jsonify({
            'success': True,
            'data': report
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar relatório: {str(e)}'}), 500

