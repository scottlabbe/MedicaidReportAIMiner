import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Setup base for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with the base class
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", os.urandom(24))

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Maximum content length for file uploads (50MB)
# Note: PDFs are now processed entirely in memory without being saved to disk
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Initialize the app with the db extension
db.init_app(app)

# Configure OpenAI API key
app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
app.config['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY')

with app.app_context():
    # Import models
    from models import Report, Finding, Recommendation, Objective, Keyword, AIProcessingLog, ScrapingQueue, SearchHistory, DuplicateCheck
    
    # Create all database tables
    db.create_all()
    logging.info("Database tables created")
