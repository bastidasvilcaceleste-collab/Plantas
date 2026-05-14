import os
from flask import Flask, render_template
from extensions import db, login_manager, migrate
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    migrate.init_app(app, db)

    from routes.auth_routes import auth_bp
    from routes.analisis_routes import analisis_bp
    from routes.settings_routes import settings_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(analisis_bp)
    app.register_blueprint(settings_bp)

    from models.database import Usuario

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(Usuario, int(user_id))

    @app.route('/')
    def index():
        return render_template('index.html')

    return app


app = create_app()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        try:
            from sqlalchemy import text
            db.session.execute(text('ALTER TABLE analisis ADD COLUMN imagen TEXT'))
            db.session.commit()
        except Exception:
            pass
    app.run(debug=True)
