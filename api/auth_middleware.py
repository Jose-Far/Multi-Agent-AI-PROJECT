from functools import wraps
from flask import request, jsonify, g
from services.auth_service import AuthService

def get_current_user():
    """Extract and validate the authenticated user from the request."""
    token = None
    
    # 1. Check Authorization header: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif auth_header:
        token = auth_header.strip()

    # 2. Check X-Session-Token header fallback
    if not token:
        token = request.headers.get("X-Session-Token")

    # 3. Check Cookie fallback
    if not token:
        token = request.cookies.get("session_token")

    if not token:
        return None

    auth_svc = AuthService()
    return auth_svc.validate_session(token)

def login_required(f):
    """Ensure the request has a valid authenticated session."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({
                "status": "error",
                "error": "UNAUTHORIZED",
                "message": "Authentication required. Please log in."
            }), 401

        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function

def roles_required(*allowed_roles):
    """Ensure the authenticated user possesses at least one of the allowed roles."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({
                    "status": "error",
                    "error": "UNAUTHORIZED",
                    "message": "Authentication required. Please log in."
                }), 401

            if user.get("role") not in allowed_roles:
                return jsonify({
                    "status": "error",
                    "error": "FORBIDDEN",
                    "message": f"Access denied. Required role: {', '.join(allowed_roles)} (Your role: {user.get('role')})"
                }), 403

            g.current_user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def permission_required(permission_name):
    """Ensure the authenticated user has the specific permission."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({
                    "status": "error",
                    "error": "UNAUTHORIZED",
                    "message": "Authentication required. Please log in."
                }), 401

            user_permissions = set(user.get("permissions", []))
            if permission_name not in user_permissions:
                return jsonify({
                    "status": "error",
                    "error": "FORBIDDEN",
                    "message": f"Access denied. Missing required permission: '{permission_name}'"
                }), 403

            g.current_user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator
