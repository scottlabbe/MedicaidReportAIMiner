from datetime import datetime
from app import db
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, Date, ForeignKey, DateTime
from sqlalchemy.orm import relationship

class Report(db.Model):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    report_title = Column(String(255), nullable=False)
    audit_organization = Column(String(255), nullable=False)
    publication_year = Column(Integer, nullable=False)
    publication_month = Column(Integer, nullable=False)
    publication_day = Column(Integer)
    overall_conclusion = Column(Text)
    llm_insight = Column(Text)
    potential_objective_summary = Column(Text)
    original_report_source_url = Column(String(255))
    state = Column(String(2))
    audit_scope = Column(Text)
    
    # PDF and system metadata
    original_filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    pdf_storage_path = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    featured = Column(Boolean, default=False)
    status = Column(String(50), default='processing')
    processing_status = Column(String(50), default='pending')
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    findings = relationship("Finding", back_populates="report", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="report", cascade="all, delete-orphan")
    objectives = relationship("Objective", back_populates="report", cascade="all, delete-orphan")
    keywords = relationship("Keyword", back_populates="report", cascade="all, delete-orphan")
    ai_logs = relationship("AIProcessingLog", back_populates="report", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Report {self.report_title}>"


class Finding(db.Model):
    __tablename__ = 'findings'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=False)
    finding_text = Column(Text, nullable=False)
    financial_impact = Column(Float)
    
    # Relationships
    report = relationship("Report", back_populates="findings")
    
    def __repr__(self):
        return f"<Finding {self.id} for Report {self.report_id}>"


class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=False)
    recommendation_text = Column(Text, nullable=False)
    related_finding_id = Column(Integer, ForeignKey('findings.id'))
    
    # Relationships
    report = relationship("Report", back_populates="recommendations")
    
    def __repr__(self):
        return f"<Recommendation {self.id} for Report {self.report_id}>"


class Objective(db.Model):
    __tablename__ = 'objectives'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=False)
    objective_text = Column(Text, nullable=False)
    
    # Relationships
    report = relationship("Report", back_populates="objectives")
    
    def __repr__(self):
        return f"<Objective {self.id} for Report {self.report_id}>"


class Keyword(db.Model):
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=False)
    keyword_text = Column(String(100), nullable=False)
    
    # Relationships
    report = relationship("Report", back_populates="keywords")
    
    def __repr__(self):
        return f"<Keyword {self.keyword_text} for Report {self.report_id}>"


class AIProcessingLog(db.Model):
    __tablename__ = 'ai_processing_logs'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    input_cost = Column(Float)
    output_cost = Column(Float)
    total_cost = Column(Float)
    processing_time_ms = Column(Integer)
    extraction_status = Column(String(50))
    error_details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    report = relationship("Report", back_populates="ai_logs")
    
    def __repr__(self):
        return f"<AIProcessingLog {self.id} for Report {self.report_id}>"
