from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime, timedelta
from src.models.transaction import Transaction, UploadHistory, ReconciliationRecord, db
from src.models.company_financial import CompanyFinancial
from sqlalchemy import func, or_
from src.constants.financial import normalize_company_financial_category, get_friendly_category_label
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService, ReconciliationConfig
from src.services.user_preference_service import UserPreferenceService
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.error_handler import handle_errors, with_resource_check
from src.utils.exceptions import (
    ValidationError, FileProcessingError, InvalidFileFormatError,
    FileSizeExceededError, DuplicateFileError, InsufficientDataError
)
from src.utils.validators import validate_pagination_params, validate_file_upload as validate_file_content
from src.utils.validation_middleware import (
    validate_file_upload, validate_input_fields, validate_financial_data,
    rate_limit, require_content_type, sanitize_path_params
)
from collections import defaultdict

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

transactions_bp = Blueprint('transactions', __name__)

# Configuracoes de upload
ALLOWED_EXTENSIONS = {'ofx', 'qfx'}
ALLOWED_XLSX_EXTENSIONS = {'xlsx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_xlsx_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_XLSX_EXTENSIONS

@transactions_bp.route('/upload-ofx', methods=['POST'])
@handle_errors
@with_resource_check(memory_limit=95)
@rate_limit(max_requests=50, window_minutes=60)  # Limit file uploads
@validate_file_upload(['ofx', 'qfx'], max_size_mb=16)
def upload_ofx():
    """
    Endpoint para upload e processamento de arquivos OFX
    """
    logger.info("OFX upload request received")
    
    # Validate request
    if 'file' not in request.files:
        logger.warning("Upload request without file")
        raise ValidationError("Nenhum arquivo enviado", user_message="Nenhum arquivo foi enviado.")
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Upload request with empty filename")
        raise ValidationError("Nenhum arquivo selecionado", user_message="Nenhum arquivo foi selecionado.")
    
    if not allowed_file(file.filename):
        logger.warning(f"Upload request with invalid file type: {file.filename}")
        raise InvalidFileFormatError(file.filename, ['ofx', 'qfx'])
        
    # Salva o arquivo temporariamente
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        file.save(temp_path)
        
        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise FileSizeExceededError(filename, file_size, MAX_FILE_SIZE)
        
        logger.info(f"OFX file uploaded: {filename} ({file_size} bytes)")
        
        # Additional file validation
        validation_result = validate_file_content(temp_path, filename, 'ofx')
        if not validation_result.is_valid:
            raise FileProcessingError(f"File validation failed: {', '.join(validation_result.errors)}", filename)
        
        # Verifica se o arquivo é duplicado
        file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
        audit_logger.log_file_upload(filename, 'OFX', file_size=file_size, file_hash=file_hash)
        
        existing_file = DuplicateDetectionService.check_file_duplicate(file_hash)
        
        if existing_file:
            logger.warning(f"Duplicate file detected: {filename} (hash: {file_hash})")
            audit_logger.log_duplicate_detection('file_level', 1, {
                'filename': filename,
                'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
            })
            
            raise DuplicateFileError(filename, original_upload_date=existing_file.upload_date.isoformat() if existing_file.upload_date else None)
        
        # Processa o arquivo OFX
        processor = OFXProcessor()
        ai_service = AIService()
        
        logger.info(f"Starting OFX processing for file: {filename}")
        
        # Parse do OFX
        ofx_data = processor.parse_ofx_file(temp_path)
        logger.info(f"OFX parsed successfully. Bank: {ofx_data['bank_name']}, Transactions: {len(ofx_data['transactions'])}")
        
        # Valida as transações
        valid_transactions = processor.validate_transactions(ofx_data['transactions'])
        logger.info(f"Transaction validation completed. Valid transactions: {len(valid_transactions)}")
        
        if not valid_transactions:
            raise InsufficientDataError('OFX processing', 1, 0)
        
        # Detecta duplicatas usando o processor
        duplicates = processor.detect_duplicates(valid_transactions)
        duplicate_count = len(duplicates)
        if duplicate_count > 0:
            logger.info(f"Duplicate transactions detected: {duplicate_count}")
            audit_logger.log_duplicate_detection('transaction_level', duplicate_count)
        
        # Aplica IA para categorização e detecção de anomalias
        logger.info("Starting AI categorization and anomaly detection")
        categorized_transactions = ai_service.categorize_transactions_batch(valid_transactions)
        analyzed_transactions = ai_service.detect_anomalies(categorized_transactions)
        audit_logger.log_ai_operation('categorization_and_anomaly_detection', len(valid_transactions), True)
        
        # Salva as transações no banco de dados
        saved_count = 0
        total_entries_processed = len(analyzed_transactions)
        duplicate_entries_details = []
        
        for transaction_data in analyzed_transactions:
            # Verifica se a transação já existe (evita duplicatas) usando o serviço unificado
            account_id = ofx_data['account_info'].get('account_id', '')
            is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
                account_id,
                transaction_data['date'],
                transaction_data['amount'],
                transaction_data['description']
            )
            
            if not is_duplicate:
                # Parse timestamp if present (string -> datetime)
                ts_val = transaction_data.get('timestamp')
                ts_dt = None
                try:
                    if isinstance(ts_val, str):
                        ts_dt = datetime.fromisoformat(ts_val)
                    elif hasattr(ts_val, 'isoformat'):
                        ts_dt = ts_val
                except Exception:
                    ts_dt = None
                transaction = Transaction(
                    bank_name=ofx_data['bank_name'],
                    account_id=account_id,
                    transaction_id=transaction_data.get('transaction_id', ''),
                    timestamp=ts_dt,
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
            else:
                # Adiciona detalhes da duplicata para o response (garantindo tipos serializáveis em JSON)
                dup_date = transaction_data.get('date')
                if hasattr(dup_date, 'isoformat'):
                    dup_date = dup_date.isoformat()
                duplicate_entries_details.append({
                    'date': dup_date,
                    'amount': float(transaction_data.get('amount', 0)),
                    'description': transaction_data.get('description')
                })
        
        # Salva o histórico de upload
        upload_record = UploadHistory(
            filename=filename,
            bank_name=ofx_data['bank_name'],
            transactions_count=saved_count,
            status='success',
            file_hash=file_hash,
            duplicate_files_count=0,  # This file is not a duplicate
            duplicate_entries_count=duplicate_count,
            total_entries_processed=total_entries_processed
        )
        db.session.add(upload_record)
        
        db.session.commit()
        
        logger.info(f"OFX processing completed successfully. Saved {saved_count} transactions")
        audit_logger.log_file_processing_result(filename, True, saved_count, duplicate_count)
        audit_logger.log_database_operation('insert', 'transactions', saved_count, True)
        
        return jsonify({
            'success': True,
            'message': f'Arquivo processado com sucesso! {saved_count} transações importadas.',
            'data': {
                'bank_name': ofx_data['bank_name'],
                'account_info': ofx_data['account_info'],
                'items_imported': saved_count,
                'items_incomplete': 0,
                'duplicates_found': duplicate_count,
                'file_duplicate': False,
                'incomplete_items': [],
                'summary': ofx_data['summary'],
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': duplicate_count,
                        'details': duplicate_entries_details
                    }
                },
                # Opcional: prévia de 5 transações processadas (para auditoria/insights no frontend)
                'transactions_preview': [
                    {
                        'date': (t.get('date').isoformat() if hasattr(t.get('date'), 'isoformat') else t.get('date')),
                        'timestamp': t.get('timestamp'),
                        'amount': float(t.get('amount', 0)),
                        'description': t.get('description'),
                        'category': t.get('category'),
                        # Cast to native Python bool to avoid numpy.bool_ serialization issues
                        'is_anomaly': bool(t.get('is_anomaly', False)),
                        'transaction_type': t.get('transaction_type')
                    }
                    for t in analyzed_transactions[:5]
                ]
            }
        })
        
    finally:
        # Always clean up temporary files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up temporary files: {str(cleanup_error)}")

@transactions_bp.route('/transactions', methods=['GET'])
@handle_errors
@rate_limit(max_requests=200, window_minutes=60)  # Higher limit for read operations
@validate_input_fields('bank', 'category', 'type', 'start_date', 'end_date')
def get_transactions():
    """
    Endpoint para listar transações com filtros

    Parâmetros aceitos (query string):
    - page: número da página (default: 1)
    - per_page: itens por página (default: 100)
    - limit: alias para per_page (para compatibilidade). Use "all" para retornar tudo
    - bank: filtra por nome do banco (igualdade)
    - category: filtra por categoria (igualdade)
    - type: 'credit' ou 'debit'
    - start_date: ISO date (YYYY-MM-DD)
    - end_date: ISO date (YYYY-MM-DD)
    - q: termo de busca textual na descrição (case-insensitive)
    - category_q: termo de busca textual na categoria (case-insensitive)
    """
    # Validate pagination parameters
    page_arg = request.args.get('page', 1)
    per_page_arg_raw = request.args.get('per_page', None)
    limit_arg = request.args.get('limit', None)
    per_page_arg = per_page_arg_raw or limit_arg

    fetch_all = isinstance(per_page_arg, str) and per_page_arg.lower() == 'all'

    # Parâmetros de filtro
    bank_name = request.args.get('bank')
    category = request.args.get('category')
    transaction_type = request.args.get('type')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    q = request.args.get('q')
    category_q = request.args.get('category_q')

    # Valida paginação com limite maior quando em modo busca (q/category_q)
    is_search = bool((q and q.strip()) or (category_q and category_q.strip()))
    max_per_page = 1000 if is_search else 100

    if fetch_all:
        pagination_params = {'page': 1, 'per_page': None}
    else:
        requested_per_page = per_page_arg or 100
        pagination_params = validate_pagination_params(page_arg, requested_per_page, max_per_page=max_per_page)

    # Validate date parameters
    start_date_obj = None
    end_date_obj = None

    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")

    # Constrói a query
    query = Transaction.query

    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)

    if category:
        query = query.filter(Transaction.category == category)

    if transaction_type:
        if transaction_type not in ['credit', 'debit']:
            raise ValidationError("transaction_type must be 'credit' or 'debit'")
        query = query.filter(Transaction.transaction_type == transaction_type)

    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())

    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())

    # Filtros textuais (descrição e categoria) - usar OR quando ambos presentes
    q_norm = q.strip().lower() if q else None
    cq_norm = category_q.strip().lower() if category_q else None

    text_conditions = []
    if q_norm:
        text_conditions.append(func.lower(Transaction.description).like(f"%{q_norm}%"))
    if cq_norm:
        text_conditions.append(func.lower(Transaction.category).like(f"%{cq_norm}%"))

    if len(text_conditions) == 1:
        query = query.filter(text_conditions[0])
    elif len(text_conditions) > 1:
        query = query.filter(or_(*text_conditions))

    # Ordena por data (mais recente primeiro)
    query = query.order_by(Transaction.date.desc())

    # Get total count before applying pagination
    total_count = query.count()

    # Aplica paginação (ou retorna tudo quando limit=all)
    if fetch_all:
        transactions = query.all()
        current_page = 1
        per_page_value = total_count
        total_pages = 1 if total_count > 0 else 0
    else:
        offset = (pagination_params['page'] - 1) * pagination_params['per_page']
        limit = pagination_params['per_page']
        transactions = query.offset(offset).limit(limit).all()
        current_page = pagination_params['page']
        per_page_value = pagination_params['per_page']
        total_pages = (total_count + per_page_value - 1) // per_page_value if per_page_value else 0

    # Enrich transactions with reconciliation and adjustment info
    tx_ids = [t.id for t in transactions]
    confirmed_ids = set()
    if tx_ids:
        confirmed_rows = db.session.query(ReconciliationRecord.bank_transaction_id).\
            filter(ReconciliationRecord.bank_transaction_id.in_(tx_ids)).\
            filter(ReconciliationRecord.status == 'confirmed').all()
        confirmed_ids = {row[0] for row in confirmed_rows}

    enriched = []
    for t in transactions:
        td = t.to_dict()
        td['is_reconciled'] = t.id in confirmed_ids
        td['was_adjusted'] = bool(getattr(t, 'justificativa', None) and str(t.justificativa).strip())
        enriched.append(td)

    if q or category_q:
        logger.info(
            f"Retrieved {len(transactions)} transactions (total: {total_count}) with search "
            f"q='{q}' category_q='{category_q}'"
        )
    else:
        logger.info(f"Retrieved {len(transactions)} transactions (total: {total_count})")

    return jsonify({
        'success': True,
        'data': {
            'transactions': enriched,
            'total_count': total_count,
            'page': current_page,
            'per_page': 'all' if fetch_all else per_page_value,
            'total_pages': total_pages
        }
    })

@transactions_bp.route('/transactions/bulk', methods=['DELETE'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
def bulk_delete_transactions():
    """
    Exclui múltiplas transações em lote.
    Espera um JSON no corpo com o formato: { "ids": [1,2,3] }
    Também remove registros de reconciliação associados às transações.
    """
    data = request.get_json() or {}
    ids = data.get('ids', [])

    if not isinstance(ids, list) or not ids:
        raise ValidationError("A list of transaction IDs is required")

    # Garantir que todos os IDs sejam inteiros
    try:
        ids = [int(x) for x in ids]
    except Exception:
        raise ValidationError("All transaction IDs must be integers")

    # Remove registros dependentes primeiro (reconciliações)
    reconciliations_deleted = db.session.query(ReconciliationRecord).\
        filter(ReconciliationRecord.bank_transaction_id.in_(ids)).delete(synchronize_session=False)

    # Remove as transações
    deleted = db.session.query(Transaction).\
        filter(Transaction.id.in_(ids)).delete(synchronize_session=False)

    db.session.commit()

    # Auditoria
    try:
        audit_logger.log_database_operation('delete', 'transactions', deleted, True)
    except Exception:
        pass

    return jsonify({
        'success': True,
        'message': f'{deleted} transações excluídas com sucesso',
        'data': {
            'deleted_count': deleted,
            'reconciliations_deleted': reconciliations_deleted
        }
    })

@transactions_bp.route('/insights', methods=['GET'])
@handle_errors
def get_insights():
    """
    Endpoint para obter insights das transações
    """
    # Parâmetros de filtro (mesmo que get_transactions)
    bank_name = request.args.get('bank')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói a query
    query = Transaction.query
    
    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)
    
    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())
    
    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())
    
    transactions = query.all()
    
    if not transactions:
        raise InsufficientDataError('insights generation', 1, 0)
    
    # Converte para formato de dicionário
    transactions_data = [t.to_dict() for t in transactions]
    
    # Gera insights
    ai_service = AIService()
    insights = ai_service.generate_insights(transactions_data)
    
    logger.info(f"Generated insights for {len(transactions)} transactions")
    
    return jsonify({
        'success': True,
        'data': insights
    })

@transactions_bp.route('/ai-insights', methods=['GET'])
@handle_errors
def get_ai_insights():
    """
    Endpoint para obter insights gerados por IA
    """
    # Parâmetros de filtro
    bank_name = request.args.get('bank')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói a query (mesmo que get_insights)
    query = Transaction.query
    
    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)
    
    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())
    
    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())
    
    transactions = query.all()
    
    if not transactions:
        raise InsufficientDataError('AI insights generation', 1, 0)
    
    # Converte para formato de dicionário
    transactions_data = [t.to_dict() for t in transactions]
    
    # Gera insights com IA
    ai_service = AIService()
    ai_insights = ai_service.generate_ai_insights(transactions_data)
    
    logger.info(f"Generated AI insights for {len(transactions)} transactions")
    
    return jsonify({
        'success': True,
        'data': {
            'ai_insights': ai_insights
        }
    })

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
                        'name': get_friendly_category_label(cat.category),
                        'raw_name': cat.category,
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
@handle_errors
@with_resource_check(memory_limit=95)
@rate_limit(max_requests=50, window_minutes=60)  # Limit file uploads
@validate_file_upload(['xlsx'], max_size_mb=10)
def upload_xlsx():
    """
    Endpoint para upload e processamento de arquivos XLSX
    """
    try:
        logger.info("XLSX upload request received")
        
        if 'file' not in request.files:
            logger.warning("XLSX upload request without file")
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("XLSX upload request with empty filename")
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_xlsx_file(file.filename):
            logger.warning(f"XLSX upload request with invalid file type: {file.filename}")
            return jsonify({'error': 'Tipo de arquivo não permitido. Use apenas .xlsx'}), 400
        
        # Salva o arquivo temporariamente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Log file upload
        file_size = os.path.getsize(temp_path)
        logger.info(f"XLSX file uploaded: {filename} ({file_size} bytes)")
        
        # Verifica se o arquivo é duplicado
        file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
        audit_logger.log_file_upload(filename, 'XLSX', file_size=file_size, file_hash=file_hash)
        
        existing_file = DuplicateDetectionService.check_file_duplicate(file_hash)
        
        if existing_file:
            logger.warning(f"Duplicate XLSX file detected: {filename} (hash: {file_hash})")
            audit_logger.log_duplicate_detection('file_level', 1, {
                'filename': filename,
                'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
            })
            
            # Remove o arquivo temporário
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return jsonify({
                'success': False,
                'message': 'Arquivo já foi processado anteriormente.',
                'data': {
                    'items_imported': 0,
                    'items_incomplete': 0,
                    'duplicates_found': 0,
                    'file_duplicate': True,
                    'incomplete_items': [],
                    'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None,
                    'duplicate_details': {
                        'file_level': {
                            'is_duplicate': True,
                            'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
                        },
                        'entry_level': {
                            'count': 0,
                            'details': []
                        }
                    }
                }
            })
        
        # Processa o arquivo XLSX
        logger.info(f"Starting XLSX processing for file: {filename}")
        processor = XLSXProcessor()
        financial_data = processor.parse_xlsx_file(temp_path)
        logger.info(f"XLSX parsed successfully. Entries found: {len(financial_data)}")

        # Enriquecimento por IA: categorização das entradas
        try:
            if financial_data:
                ai_service = AIService()
                logger.info("Starting AI categorization for XLSX entries")
                financial_data = ai_service.categorize_transactions_batch(financial_data)
                logger.info("AI categorization completed for XLSX entries")
        except Exception as ai_err:
            # Não interrompe o fluxo em caso de falha na IA; mantém categorias originais
            logger.warning(f"AI categorization failed for XLSX entries: {str(ai_err)}")
        
        # Salva os dados no banco de dados
        saved_count = 0
        incomplete_entries = []
        errors = []
        duplicate_entries_details = []
        duplicate_count = 0
        type_conflict_details = []
        type_conflict_count = 0
        total_entries_processed = len(financial_data)
        
        for i, entry_data in enumerate(financial_data):
            try:
                # Verifica se a data é válida
                if entry_data['date'] is None:
                    # Adiciona à lista de entradas incompletas em vez de pular
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = "Data inválida ou ausente"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Verifica se a descrição está vazia
                if not entry_data['description'] or entry_data['description'].strip() == '':
                    # Adiciona à lista de entradas incompletas em vez de pular
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = "Descrição ausente"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Verifica se a entrada já existe (evita duplicatas)
                try:
                    is_duplicate = DuplicateDetectionService.check_financial_entry_duplicate(
                        entry_data['date'],
                        entry_data['amount'],
                        entry_data['description']
                    )
                    
                    if is_duplicate:
                        # Adiciona detalhes da duplicata para o response
                        duplicate_entries_details.append({
                            'date': entry_data['date'].isoformat() if entry_data['date'] else None,
                            'amount': entry_data['amount'],
                            'description': entry_data['description'],
                            'row_number': i + 1
                        })
                        duplicate_count += 1
                        continue
                        
                except Exception as duplicate_error:
                    # Se falhar a verificação de duplicata, trata como erro e não insere
                    error_msg = f"Erro na verificação de duplicata na linha {i + 1}: {str(duplicate_error)}"
                    errors.append(error_msg)
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = f"Erro na verificação de duplicata: {str(duplicate_error)}"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Registrar conflito de tipo (apenas informativo)
                try:
                    if entry_data.get('transaction_type_conflict'):
                        type_conflict_count += 1
                        type_conflict_details.append({
                            'row_number': i + 1,
                            'date': entry_data['date'].isoformat() if entry_data['date'] else None,
                            'amount': entry_data.get('amount'),
                            'description': entry_data.get('description'),
                            'label': entry_data.get('tipo_raw') or entry_data.get('transaction_type_label'),
                            'final_type': entry_data.get('transaction_type')
                        })
                except Exception:
                    pass

                # Se chegou até aqui, a entrada é válida e não é duplicata
                financial_entry = CompanyFinancial(
                    date=entry_data['date'],
                    description=entry_data['description'],
                    amount=entry_data['amount'],
                    category=normalize_company_financial_category(entry_data.get('category')),
                    cost_center=entry_data['cost_center'],
                    department=entry_data['department'],
                    project=entry_data['project'],
                    transaction_type=entry_data['transaction_type']
                )
                db.session.add(financial_entry)
                saved_count += 1
                
            except Exception as e:
                # Adiciona erro específico para esta entrada
                error_msg = f"Erro na linha {i + 1}: {str(e)}"
                errors.append(error_msg)
                entry_data['row_number'] = i + 1
                entry_data['error'] = str(e)
                incomplete_entries.append(entry_data)
        
        # Salva o histórico de upload
        upload_record = UploadHistory(
            filename=filename,
            bank_name='xlsx_file',
            transactions_count=saved_count,
            status='success',
            file_hash=file_hash,
            duplicate_files_count=0,  # This file is not a duplicate
            duplicate_entries_count=duplicate_count,
            total_entries_processed=total_entries_processed
        )
        db.session.add(upload_record)
        
        db.session.commit()
        
        logger.info(f"XLSX processing completed. Saved: {saved_count}, Incomplete: {len(incomplete_entries)}, Duplicates: {duplicate_count}")
        audit_logger.log_file_processing_result(filename, True, saved_count, duplicate_count, errors)
        audit_logger.log_database_operation('insert', 'company_financial', saved_count, True)
        
        if duplicate_count > 0:
            audit_logger.log_duplicate_detection('entry_level', duplicate_count, {
                'details': duplicate_entries_details[:10]  # Log first 10 duplicates
            })
        
        # Remove o arquivo temporário
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Prepara a mensagem de retorno
        message = f'Arquivo processado com sucesso! {saved_count} entradas importadas.'
        if len(incomplete_entries) > 0:
            message += f' {len(incomplete_entries)} entradas incompletas encontradas e requerem correção.'
        if duplicate_count > 0:
            message += f' {duplicate_count} entradas duplicadas foram ignoradas.'
        
        response_data = {
            'success': True,
            'message': message,
            'data': {
                'items_imported': saved_count,
                'items_incomplete': len(incomplete_entries),
                'duplicates_found': duplicate_count,
                'file_duplicate': False,
                'incomplete_items': incomplete_entries,
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': duplicate_count,
                        'details': duplicate_entries_details
                    }
                },
                'type_conflicts': {
                    'count': type_conflict_count,
                    'details': type_conflict_details[:20]
                }
            }
        }
        
        # Adiciona detalhes de erros se houver
        if errors:
            response_data['errors'] = errors
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing XLSX file {filename}: {str(e)}", exc_info=True)
        audit_logger.log_file_processing_result(filename, False, 0, 0, [str(e)])
        
        # Remove o arquivo temporário em caso de erro
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500

@transactions_bp.route('/upload-xlsx-corrected', methods=['POST'])
@rate_limit(max_requests=30, window_minutes=60)
@require_content_type('application/json')
@validate_financial_data('company_financial')
def upload_xlsx_corrected():
    """
    Endpoint para upload e processamento de dados corrigidos
    """
    try:
        # Obtém os dados corrigidos do corpo da requisição
        data = request.get_json()
        
        if not data or 'entries' not in data:
            return jsonify({'error': 'Nenhum dado enviado'}), 400
        
        corrected_entries = data['entries']
        
        # Salva os dados corrigidos no banco de dados
        saved_count = 0
        errors = []
        
        for i, entry_data in enumerate(corrected_entries):
            try:
                # Valida os dados corrigidos
                if not entry_data.get('date'):
                    errors.append(f"Entrada {i+1}: Data ausente")
                    continue
                
                if not entry_data.get('description') or entry_data['description'].strip() == '':
                    errors.append(f"Entrada {i+1}: Descrição ausente")
                    continue
                
                # Converte a data para o formato correto
                if isinstance(entry_data['date'], str):
                    from datetime import datetime
                    entry_data['date'] = datetime.fromisoformat(entry_data['date'].replace('Z', '+00:00')).date()
                
                financial_entry = CompanyFinancial(
                    date=entry_data['date'],
                    description=entry_data['description'],
                    amount=float(entry_data.get('amount', 0)),
                    category=normalize_company_financial_category(entry_data.get('category')),
                    cost_center=entry_data.get('cost_center', ''),
                    department=entry_data.get('department', ''),
                    project=entry_data.get('project', ''),
                    transaction_type=entry_data.get('transaction_type', 'expense'),
                    justificativa=entry_data.get('justificativa')
                )
                db.session.add(financial_entry)
                saved_count += 1
            except Exception as e:
                errors.append(f"Entrada {i+1}: Erro ao processar - {str(e)}")
        
        db.session.commit()
        
        # Prepara a mensagem de retorno
        message = f'Dados corrigidos processados com sucesso! {saved_count} entradas importadas.'
        if errors:
            message += f' {len(errors)} entradas com erros.'
        
        response_data = {
            'success': True,
            'message': message,
            'data': {
                'items_imported': saved_count,
                'items_incomplete': 0,
                'duplicates_found': 0,
                'file_duplicate': False,
                'incomplete_items': [],
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': 0,
                        'details': []
                    }
                }
            }
        }
        
        # Adiciona detalhes de erros se houver
        if errors:
            response_data['errors'] = errors
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Erro ao processar dados corrigidos: {str(e)}'}), 500

# Update a bank transaction (adjust fields and justificativa)
@transactions_bp.route('/transactions/<int:transaction_id>', methods=['PUT'])
@handle_errors
@rate_limit(max_requests=200, window_minutes=60)
@require_content_type('application/json')
@sanitize_path_params('transaction_id')
def update_transaction(transaction_id):
    data = request.get_json() or {}

    tx = Transaction.query.get(transaction_id)
    if not tx:
        return jsonify({'error': 'Transação não encontrada'}), 404

    # Update allowed fields if provided
    try:
        if 'date' in data and data['date']:
            # Accept ISO string or date string YYYY-MM-DD
            if isinstance(data['date'], str):
                tx.date = datetime.fromisoformat(data['date'].replace('Z', '+00:00')).date()
        if 'amount' in data and data['amount'] is not None:
            tx.amount = float(data['amount'])
        if 'description' in data:
            tx.description = str(data['description']).strip()
        if 'category' in data:
            tx.category = str(data['category']).strip() if data['category'] is not None else None
        if 'transaction_type' in data and data['transaction_type'] in ['debit', 'credit']:
            tx.transaction_type = data['transaction_type']
        if 'is_anomaly' in data:
            tx.is_anomaly = bool(data['is_anomaly'])
        if 'justificativa' in data:
            tx.justificativa = str(data['justificativa']).strip() if data['justificativa'] else None

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Transação atualizada com sucesso',
            'data': tx.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Erro ao atualizar transação: {str(e)}'}), 500


# Update a company financial entry (adjust fields and justificativa)
@transactions_bp.route('/company-financial/<int:entry_id>', methods=['PUT'])
@handle_errors
@rate_limit(max_requests=200, window_minutes=60)
@require_content_type('application/json')
@sanitize_path_params('entry_id')
def update_company_financial(entry_id):
    data = request.get_json() or {}

    entry = CompanyFinancial.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entrada financeira não encontrada'}), 404

    try:
        if 'date' in data and data['date']:
            if isinstance(data['date'], str):
                entry.date = datetime.fromisoformat(data['date'].replace('Z', '+00:00')).date()
        if 'amount' in data and data['amount'] is not None:
            entry.amount = float(data['amount'])
        if 'description' in data:
            entry.description = str(data['description']).strip()
        if 'category' in data:
            entry.category = normalize_company_financial_category(data['category'])
        if 'cost_center' in data:
            entry.cost_center = str(data['cost_center']).strip() if data['cost_center'] is not None else None
        if 'department' in data:
            entry.department = str(data['department']).strip() if data['department'] is not None else None
        if 'project' in data:
            entry.project = str(data['project']).strip() if data['project'] is not None else None
        if 'transaction_type' in data and data['transaction_type'] in ['expense', 'income']:
            entry.transaction_type = data['transaction_type']
        if 'justificativa' in data:
            entry.justificativa = str(data['justificativa']).strip() if data['justificativa'] else None

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Entrada financeira atualizada com sucesso',
            'data': entry.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Erro ao atualizar entrada financeira: {str(e)}'}), 500

@transactions_bp.route('/company-financial/<int:entry_id>', methods=['DELETE'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('entry_id')
def delete_company_financial(entry_id):
    """
    Endpoint para excluir uma entrada financeira da empresa
    """
    entry = CompanyFinancial.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entrada financeira não encontrada'}), 404

    # Impede exclusão se houver registros de reconciliação associados
    linked_count = db.session.query(ReconciliationRecord).\
        filter(ReconciliationRecord.company_entry_id == entry_id).count()
    if linked_count > 0:
        return jsonify({
            'error': 'Não é possível excluir: entrada vinculada à reconciliação',
            'details': [f'{linked_count} registro(s) de reconciliação associados']
        }), 400

    try:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'Entrada financeira excluída com sucesso',
            'data': {'id': entry_id}
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Erro ao excluir entrada financeira: {str(e)}'}), 500

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

        # Enrich with reconciliation and adjustment info
        entry_ids = [e.id for e in entries]
        confirmed_ids = set()
        if entry_ids:
            confirmed_rows = db.session.query(ReconciliationRecord.company_entry_id).\
                filter(ReconciliationRecord.company_entry_id.in_(entry_ids)).\
                filter(ReconciliationRecord.status == 'confirmed').all()
            confirmed_ids = {row[0] for row in confirmed_rows}

        enriched = []
        for e in entries:
            ed = e.to_dict()
            ed['is_reconciled'] = e.id in confirmed_ids
            ed['was_adjusted'] = bool(getattr(e, 'justificativa', None) and str(e.justificativa).strip())
            enriched.append(ed)

        return jsonify({
            'success': True,
            'data': {
                'entries': enriched,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar entradas financeiras: {str(e)}'}), 500

@transactions_bp.route('/company-financial', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_financial_data('company_financial')
def create_company_financial():
    """
    Endpoint para criar uma nova entrada financeira da empresa manualmente
    """
    data = request.get_json(silent=True) or {}

    try:
        # Parse and normalize fields
        date_val = data.get('date')
        if isinstance(date_val, str) and date_val:
            # Accept ISO date or datetime strings
            date_obj = datetime.fromisoformat(date_val.replace('Z', '+00:00')).date()
        elif isinstance(date_val, datetime):
            date_obj = date_val.date()
        else:
            date_obj = date_val  # Assume already a date

        entry = CompanyFinancial(
            date=date_obj,
            description=(data.get('description') or '').strip(),
            amount=float(data.get('amount') or 0),
            category=normalize_company_financial_category(data.get('category')),
            cost_center=(data.get('cost_center') or '').strip() or None,
            department=(data.get('department') or '').strip() or None,
            project=(data.get('project') or '').strip() or None,
            transaction_type=data.get('transaction_type') or 'expense',
            justificativa=(data.get('justificativa') or '').strip() or None
        )

        db.session.add(entry)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Entrada financeira criada com sucesso',
            'data': entry.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Erro ao criar entrada financeira: {str(e)}'}), 500

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
            category_key = normalize_company_financial_category(entry.category)
            if category_key not in category_stats:
                category_stats[category_key] = {'count': 0, 'total': 0}
            category_stats[category_key]['count'] += 1
            category_stats[category_key]['total'] += entry.amount
        
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
                'accuracy': result.get('accuracy'),
                'training_data_count': result.get('training_data_count'),
                'categories_count': result.get('categories_count'),
                'metrics': result.get('metrics', {})
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
@handle_errors
@with_resource_check
@rate_limit(max_requests=20, window_minutes=60)  # Lower limit for intensive operations
@validate_input_fields('start_date', 'end_date')
def start_reconciliation():
    """
    Endpoint para iniciar o processo de reconciliação entre transações bancárias e entradas financeiras
    """
    # Parâmetros para filtrar dados
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói queries para dados bancários e financeiros
    bank_query = Transaction.query
    company_query = CompanyFinancial.query
    
    if start_date_obj:
        bank_query = bank_query.filter(Transaction.date >= start_date_obj.date())
        company_query = company_query.filter(CompanyFinancial.date >= start_date_obj.date())
    
    if end_date_obj:
        bank_query = bank_query.filter(Transaction.date <= end_date_obj.date())
        company_query = company_query.filter(CompanyFinancial.date <= end_date_obj.date())
    
    # Obtém os dados
    bank_transactions = bank_query.all()
    company_entries = company_query.all()
    
    if not bank_transactions:
        raise InsufficientDataError('reconciliation', 1, 0)
    
    if not company_entries:
        raise InsufficientDataError('reconciliation', 1, 0)
    
    # Realiza a reconciliação
    logger.info(f"Starting reconciliation with {len(bank_transactions)} bank transactions and {len(company_entries)} company entries")
    reconciliation_service = ReconciliationService()
    matches = reconciliation_service.find_matches(bank_transactions, company_entries)
    records = reconciliation_service.create_reconciliation_records(matches)
    
    logger.info(f"Reconciliation completed. {len(records)} matches found")
    audit_logger.log_reconciliation_operation('start', len(records), 'completed', {
        'bank_transactions': len(bank_transactions),
        'company_entries': len(company_entries),
        'date_range': {'start': start_date, 'end': end_date}
    })
    
    return jsonify({
        'success': True,
        'message': f'Reconciliação iniciada com sucesso. {len(records)} correspondências encontradas.',
        'data': {
            'matches_count': len(records),
            'bank_transactions_count': len(bank_transactions),
            'company_entries_count': len(company_entries)
        }
    })

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
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def confirm_reconciliation(reconciliation_id):
    """
    Endpoint para confirmar uma reconciliação
    """
    # Validate reconciliation_id
    if reconciliation_id <= 0:
        raise ValidationError("Invalid reconciliation ID")
    
    reconciliation_service = ReconciliationService()
    success = reconciliation_service.confirm_reconciliation(reconciliation_id)
    
    logger.info(f"Reconciliation {reconciliation_id} confirmed successfully")
    
    return jsonify({
        'success': True,
        'message': 'Reconciliação confirmada com sucesso',
        'data': {
            'reconciliation_id': reconciliation_id
        }
    })

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/reject', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def reject_reconciliation(reconciliation_id):
    """
    Endpoint para rejeitar uma reconciliação
    """
    # Validate reconciliation_id
    if reconciliation_id <= 0:
        raise ValidationError("Invalid reconciliation ID")
    
    reconciliation_service = ReconciliationService()
    success = reconciliation_service.reject_reconciliation(reconciliation_id)
    
    logger.info(f"Reconciliation {reconciliation_id} rejected successfully")
    
    return jsonify({
        'success': True,
        'message': 'Reconciliação rejeitada com sucesso',
        'data': {
            'reconciliation_id': reconciliation_id
        }
    })

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

# Enhanced Reconciliation Endpoints

@transactions_bp.route('/reconciliation/configure', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('config_data')
def configure_reconciliation():
    """
    Endpoint to save user reconciliation configuration
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        user_id = data.get('user_id', 1)  # Default to user 1 if not specified
        config_name = data.get('config_name', 'default')
        config_data = data.get('config_data', {})
        is_default = data.get('is_default', False)
        
        user_preference_service = UserPreferenceService()
        
        # Check if config exists and update, or create new
        existing_config = user_preference_service.get_user_config(user_id, config_name)
        
        if existing_config:
            user_config = user_preference_service.update_user_config(
                user_id, config_name, config_data, is_default
            )
            message = 'Configuration updated successfully'
        else:
            user_config = user_preference_service.create_user_config(
                user_id, config_name, config_data, is_default
            )
            message = 'Configuration created successfully'
        
        return jsonify({
            'success': True,
            'message': message,
            'data': user_config.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error configuring reconciliation: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/configure/<int:user_id>', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('user_id')
def get_user_configurations(user_id):
    """
    Endpoint to get user reconciliation configurations
    """
    try:
        config_name = request.args.get('config_name')
        user_preference_service = UserPreferenceService()
        
        if config_name:
            user_config = user_preference_service.get_user_config(user_id, config_name)
            if not user_config:
                return jsonify({'error': 'Configuration not found'}), 404
            
            config_data = user_config.to_dict()
        else:
            # Get default config or all configs
            if request.args.get('default_only') == 'true':
                user_config = user_preference_service.get_user_config(user_id)
                config_data = user_config.to_dict() if user_config else None
            else:
                user_configs = user_preference_service.get_all_user_configs(user_id)
                config_data = [config.to_dict() for config in user_configs]
        
        return jsonify({
            'success': True,
            'data': config_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting user configurations: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/preview', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=30, window_minutes=60)  # Lower limit for intensive operations
@validate_input_fields('config_data')
def preview_reconciliation():
    """
    Endpoint to preview reconciliation matches with custom configuration
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        # Get date range
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        config_data = data.get('config_data', {})
        user_id = data.get('user_id', 1)
        
        # Validate date parameters
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
        
        # Build queries
        bank_query = Transaction.query
        company_query = CompanyFinancial.query
        
        if start_date_obj:
            bank_query = bank_query.filter(Transaction.date >= start_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date_obj:
            bank_query = bank_query.filter(Transaction.date <= end_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Get data
        bank_transactions = bank_query.all()
        company_entries = company_query.all()
        
        if not bank_transactions:
            return jsonify({
                'success': True,
                'message': 'No bank transactions found for the specified criteria',
                'data': {'matches': [], 'summary': {'total_transactions': 0, 'total_matches': 0}}
            })
        
        if not company_entries:
            return jsonify({
                'success': True,
                'message': 'No company entries found for the specified criteria',
                'data': {'matches': [], 'summary': {'total_transactions': 0, 'total_matches': 0}}
            })
        
        # Create config and run matching
        config = ReconciliationConfig()
        config.update_from_dict(config_data)
        
        # Get user-specific config if available
        user_preference_service = UserPreferenceService()
        user_config = user_preference_service.get_reconciliation_config_for_user(user_id)
        if user_config:
            config = user_config
        
        reconciliation_service = ReconciliationService(config)
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        
        # Format matches for response
        formatted_matches = []
        for match in matches:
            formatted_match = {
                'bank_transaction': match['bank_transaction'].to_dict(),
                'company_entry': match['company_entry'].to_dict(),
                'match_score': match['match_score'],
                'match_rank': match.get('match_rank', 1),
                'score_breakdown': match.get('score_breakdown', {})
            }
            formatted_matches.append(formatted_match)
        
        return jsonify({
            'success': True,
            'message': 'Reconciliation preview completed successfully',
            'data': {
                'matches': formatted_matches,
                'summary': {
                    'total_transactions': len(bank_transactions),
                    'total_entries': len(company_entries),
                    'total_matches': len(matches),
                    'config_used': config.to_dict()
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error previewing reconciliation: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/manual-match', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('bank_transaction_id', 'company_entry_id')
def create_manual_match():
    """
    Endpoint to create a manual match between bank transaction and company entry
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        bank_transaction_id = data.get('bank_transaction_id')
        company_entry_id = data.get('company_entry_id')
        user_confidence = data.get('user_confidence', 1.0)
        justification = data.get('justification')
        user_id = data.get('user_id', 1)
        
        reconciliation_service = ReconciliationService()
        manual_match = reconciliation_service.create_manual_match(
            bank_transaction_id, company_entry_id, user_confidence, justification
        )
        
        # Record user feedback
        user_preference_service = UserPreferenceService()
        user_preference_service.record_user_feedback(
            user_id=user_id,
            feedback_type='manual_match',
            reconciliation_id=manual_match.id,
            adjusted_score=user_confidence,
            confidence_rating=data.get('confidence_rating', 5),
            justification=justification,
            feedback_data={'manual_match': True, 'user_confidence': user_confidence}
        )
        
        return jsonify({
            'success': True,
            'message': 'Manual match created successfully',
            'data': manual_match.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error creating manual match: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/batch-confirm', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('reconciliation_ids')
def batch_confirm_reconciliations():
    """
    Endpoint to confirm multiple reconciliation records in batch
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        reconciliation_ids = data.get('reconciliation_ids', [])
        user_id = data.get('user_id', 1)
        
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.batch_confirm_reconciliations(reconciliation_ids)
        
        # Record user feedback for batch operation
        user_preference_service = UserPreferenceService()
        user_preference_service.record_user_feedback(
            user_id=user_id,
            feedback_type='confirm',
            feedback_data={'batch_operation': True, 'confirmed_count': result['confirmed_count']}
        )
        
        return jsonify({
            'success': True,
            'message': f'Batch confirmation completed. {result["confirmed_count"]} confirmed.',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in batch confirmation: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/batch-reject', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('reconciliation_ids')
def batch_reject_reconciliations():
    """
    Endpoint to reject multiple reconciliation records in batch
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        reconciliation_ids = data.get('reconciliation_ids', [])
        user_id = data.get('user_id', 1)
        
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.batch_reject_reconciliations(reconciliation_ids)
        
        # Record user feedback for batch operation
        user_preference_service = UserPreferenceService()
        user_preference_service.record_user_feedback(
            user_id=user_id,
            feedback_type='reject',
            feedback_data={'batch_operation': True, 'rejected_count': result['rejected_count']}
        )
        
        return jsonify({
            'success': True,
            'message': f'Batch rejection completed. {result["rejected_count"]} rejected.',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in batch rejection: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/adjust-score', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=50, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def adjust_match_score(reconciliation_id):
    """
    Endpoint to adjust the match score of a reconciliation record
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        new_score = data.get('new_score')
        justification = data.get('justification')
        user_id = data.get('user_id', 1)
        
        if new_score is None:
            raise ValidationError("New score is required")
        
        reconciliation_service = ReconciliationService()
        success = reconciliation_service.adjust_match_score(reconciliation_id, new_score, justification)
        
        if success:
            # Record user feedback
            user_preference_service = UserPreferenceService()
            user_preference_service.record_user_feedback(
                user_id=user_id,
                feedback_type='adjust',
                reconciliation_id=reconciliation_id,
                adjusted_score=new_score,
                confidence_rating=data.get('confidence_rating', 3),
                justification=justification,
                feedback_data={'score_adjustment': True, 'new_score': new_score}
            )
            
            return jsonify({
                'success': True,
                'message': 'Match score adjusted successfully',
                'data': {'reconciliation_id': reconciliation_id, 'new_score': new_score}
            })
        else:
            return jsonify({'error': 'Failed to adjust match score'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Error adjusting match score: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/<int:bank_transaction_id>/suggestions', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('bank_transaction_id')
def get_reconciliation_suggestions(bank_transaction_id):
    """
    Endpoint to get AI-powered matching suggestions for a bank transaction
    """
    try:
        limit = request.args.get('limit', 5, type=int)
        user_id = request.args.get('user_id', 1, type=int)
        
        # Get user-specific config
        user_preference_service = UserPreferenceService()
        user_config = user_preference_service.get_reconciliation_config_for_user(user_id)
        
        reconciliation_service = ReconciliationService(user_config)
        suggestions = reconciliation_service.get_reconciliation_suggestions(bank_transaction_id, limit)
        
        # Format suggestions for response
        formatted_suggestions = []
        for suggestion in suggestions:
            formatted_suggestion = {
                'company_entry': suggestion['company_entry'].to_dict(),
                'match_score': suggestion['match_score'],
                'score_breakdown': suggestion['score_breakdown'],
                'suggestion_reason': suggestion['suggestion_reason']
            }
            formatted_suggestions.append(formatted_suggestion)
        
        return jsonify({
            'success': True,
            'message': f'Found {len(formatted_suggestions)} suggestions',
            'data': {
                'bank_transaction_id': bank_transaction_id,
                'suggestions': formatted_suggestions
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting reconciliation suggestions: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/feedback/<int:user_id>', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@sanitize_path_params('user_id')
def get_user_feedback_history(user_id):
    """
    Endpoint to get user's reconciliation feedback history
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        
        user_preference_service = UserPreferenceService()
        feedback_history = user_preference_service.get_user_feedback_history(user_id, limit)
        
        return jsonify({
            'success': True,
            'data': {
                'feedback_history': [feedback.to_dict() for feedback in feedback_history]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting user feedback history: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/learning/<int:user_id>', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@sanitize_path_params('user_id')
def get_user_learning_data(user_id):
    """
    Endpoint to get aggregated learning data from user feedback
    """
    try:
        user_preference_service = UserPreferenceService()
        learning_data = user_preference_service.get_user_learning_data(user_id)
        
        return jsonify({
            'success': True,
            'data': learning_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting user learning data: {str(e)}'}), 500

@transactions_bp.route('/analyze-xlsx', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=30, window_minutes=60)  # Lower limit for analysis operations
@validate_file_upload(['xlsx'], max_size_mb=10)
def analyze_xlsx():
    """
    Endpoint para analisar arquivo XLSX e determinar tipo/conteúdo
    """
    logger.info("XLSX analysis request received")

    # Validate request
    if 'file' not in request.files:
        logger.warning("Analysis request without file")
        raise ValidationError("Nenhum arquivo enviado", user_message="Nenhum arquivo foi enviado.")

    file = request.files['file']

    if file.filename == '':
        logger.warning("Analysis request with empty filename")
        raise ValidationError("Nenhum arquivo selecionado", user_message="Nenhum arquivo foi selecionado.")

    if not allowed_xlsx_file(file.filename):
        logger.warning(f"Analysis request with invalid file type: {file.filename}")
        raise InvalidFileFormatError(file.filename, ['xlsx'])

    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        file.save(temp_path)

        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise FileSizeExceededError(filename, file_size, MAX_FILE_SIZE)

        logger.info(f"XLSX file uploaded for analysis: {filename} ({file_size} bytes)")

        # Additional file validation
        validation_result = validate_file_content(temp_path, filename, 'xlsx')
        if not validation_result.is_valid:
            raise FileProcessingError(f"File validation failed: {', '.join(validation_result.errors)}", filename)

        # Analyze the XLSX file
        logger.info(f"Starting XLSX analysis for file: {filename}")
        processor = XLSXProcessor()
        analysis_result = processor.analyze_xlsx_file(temp_path)

        logger.info(f"XLSX analysis completed successfully for file: {filename}")

        return jsonify({
            'success': True,
            'message': f'Análise do arquivo {filename} concluída com sucesso.',
            'data': analysis_result
        })

    finally:
        # Always clean up temporary files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up temporary files: {str(cleanup_error)}")

# Anomaly Detection Endpoints

@transactions_bp.route('/reconciliation/anomaly-detection', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=10, window_minutes=60)  # Lower limit for intensive operations
@validate_input_fields('start_date', 'end_date')
def process_reconciliation_with_anomaly_detection():
    """
    Endpoint para processar reconciliação com detecção de anomalias e supervisão humana
    """
    try:
        data = request.get_json() or {}
        
        # Get date range
        start_date = data.get('start_date') or request.args.get('start_date')
        end_date = data.get('end_date') or request.args.get('end_date')
        user_id = data.get('user_id', 1)
        
        # Validate date parameters
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
        
        # Validate date range
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")
        
        # Build queries for bank and financial data
        bank_query = Transaction.query
        company_query = CompanyFinancial.query
        
        if start_date_obj:
            bank_query = bank_query.filter(Transaction.date >= start_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date_obj:
            bank_query = bank_query.filter(Transaction.date <= end_date_obj.date())
            company_query = company_query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Get the data
        bank_transactions = bank_query.all()
        company_entries = company_query.all()
        
        if not bank_transactions:
            return jsonify({
                'success': True,
                'message': 'No bank transactions found for the specified criteria',
                'data': {'total_matches': 0, 'anomalies_detected': 0}
            })
        
        if not company_entries:
            return jsonify({
                'success': True,
                'message': 'No company entries found for the specified criteria',
                'data': {'total_matches': 0, 'anomalies_detected': 0}
            })
        
        # Process reconciliation with anomaly detection
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.process_reconciliation_with_anomaly_detection(
            bank_transactions, company_entries, user_id
        )
        
        return jsonify({
            'success': True,
            'message': 'Reconciliation with anomaly detection completed successfully',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing reconciliation with anomaly detection: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomalies', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
def get_anomalous_reconciliations():
    """
    Endpoint para obter lista de reconciliações anômalas com paginação e filtros
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        severity_filter = request.args.get('severity')
        status_filter = request.args.get('status')
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.get_anomalous_reconciliations(
            limit, offset, severity_filter, status_filter
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting anomalous reconciliations: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomaly/<int:reconciliation_id>/review', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def review_anomaly(reconciliation_id):
    """
    Endpoint para revisar e resolver uma anomalia com supervisão humana
    """
    try:
        data = request.get_json() or {}
        
        user_id = data.get('user_id', 1)
        action = data.get('action')
        justification = data.get('justification')
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.review_anomaly(reconciliation_id, user_id, action, justification)
        
        return jsonify({
            'success': True,
            'message': f'Anomaly review completed successfully. Action: {action}',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error reviewing anomaly: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomaly/<int:reconciliation_id>/suggestions', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def get_anomaly_workflow_suggestions(reconciliation_id):
    """
    Endpoint para obter sugestões de fluxo de trabalho com IA para resolução de anomalias
    """
    try:
        reconciliation_service = ReconciliationService()
        suggestions = reconciliation_service.get_anomaly_workflow_suggestions(reconciliation_id)
        
        return jsonify({
            'success': True,
            'data': suggestions
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting anomaly workflow suggestions: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomaly/<int:reconciliation_id>/escalate', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def escalate_anomaly(reconciliation_id):
    """
    Endpoint para escalar uma anomalia para autoridade superior ou especialista
    """
    try:
        data = request.get_json() or {}
        
        user_id = data.get('user_id', 1)
        escalation_reason = data.get('escalation_reason')
        target_user_id = data.get('target_user_id')
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.escalate_anomaly(
            reconciliation_id, user_id, escalation_reason, target_user_id
        )
        
        return jsonify({
            'success': True,
            'message': 'Anomaly escalated successfully',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error escalating anomaly: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomaly/batch-review', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
def bulk_anomaly_review():
    """
    Endpoint para revisar múltiplas anomalias em lote
    """
    try:
        data = request.get_json() or {}
        
        reconciliation_ids = data.get('reconciliation_ids', [])
        user_id = data.get('user_id', 1)
        action = data.get('action')
        justification = data.get('justification')
        
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        reconciliation_service = ReconciliationService()
        result = reconciliation_service.bulk_anomaly_review(reconciliation_ids, user_id, action, justification)
        
        return jsonify({
            'success': True,
            'message': f'Bulk anomaly review completed. {result["processed_count"]} processed.',
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in bulk anomaly review: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/anomaly/statistics', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_anomaly_statistics():
    """
    Endpoint para obter estatísticas abrangentes de anomalias
    """
    try:
        reconciliation_service = ReconciliationService()
        statistics = reconciliation_service.get_anomaly_statistics()
        
        return jsonify({
            'success': True,
            'data': statistics
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting anomaly statistics: {str(e)}'}), 500
