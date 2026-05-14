from flask import Blueprint, render_template, request, redirect, url_for, flash, session, make_response
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db
from models.database import Usuario

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/registro', methods=['GET', 'POST'])
def registro():
    if current_user.is_authenticated:
        return redirect(url_for('analisis.dashboard'))

    if request.method == 'POST':
        nombre = request.form.get('nombre', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not nombre or not email or not password:
            flash('Completa todos los campos.', 'error')
            return render_template('registro.html')

        if len(password) < 6:
            flash('La contraseña debe tener mínimo 6 caracteres.', 'error')
            return render_template('registro.html')

        if Usuario.query.filter_by(email=email).first():
            flash('El email ya está registrado.', 'error')
            return render_template('registro.html')

        usuario = Usuario(
            nombre=nombre,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(usuario)
        db.session.commit()
        login_user(usuario)
        return redirect(url_for('analisis.dashboard'))

    return render_template('registro.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('analisis.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        usuario = Usuario.query.filter_by(email=email).first()

        if not usuario or not check_password_hash(usuario.password, password):
            flash('Email o contraseña incorrectos.', 'error')
            return render_template('login.html')

        login_user(usuario, remember=True)
        return redirect(url_for('analisis.dashboard'))

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    resp = make_response(redirect(url_for('auth.login')))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
