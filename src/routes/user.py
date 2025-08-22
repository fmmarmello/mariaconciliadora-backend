from flask import Blueprint, jsonify, request
from src.models.user import User, db
from src.utils.validation_middleware import (
    validate_input_fields, rate_limit, require_content_type, sanitize_path_params
)

user_bp = Blueprint('user', __name__)

@user_bp.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@user_bp.route('/users', methods=['POST'])
@rate_limit(max_requests=10, window_minutes=60)  # Limit user creation
@require_content_type('application/json')
@validate_input_fields('username', 'email')
def create_user():
    
    data = request.json
    user = User(username=data['username'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

@user_bp.route('/users/<int:user_id>', methods=['GET'])
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('user_id')
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@user_bp.route('/users/<int:user_id>', methods=['PUT'])
@rate_limit(max_requests=20, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('username', 'email')
@sanitize_path_params('user_id')
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.json
    user.username = data.get('username', user.username)
    user.email = data.get('email', user.email)
    db.session.commit()
    return jsonify(user.to_dict())

@user_bp.route('/users/<int:user_id>', methods=['DELETE'])
@rate_limit(max_requests=10, window_minutes=60)  # Lower limit for destructive operations
@sanitize_path_params('user_id')
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return '', 204
