import logging
from datetime import datetime
from flask import current_app
from app import db
from models import Report, Finding, Recommendation, Objective, Keyword, AIProcessingLog

def check_duplicate_report(file_hash, filename):
    """
    Check if a report with the same file hash or filename already exists.
    
    Args:
        file_hash: SHA-256 hash of the file
        filename: Original filename of the report
        
    Returns:
        tuple: (is_duplicate, existing_report, reason)
    """
    # Check for exact content match (file hash)
    existing_report = Report.query.filter_by(file_hash=file_hash).first()
    if existing_report:
        return (True, existing_report, "file_hash")
    
    # Check for filename match (might be different content)
    existing_report = Report.query.filter_by(original_filename=filename).first()
    if existing_report:
        return (True, existing_report, "filename")
    
    return (False, None, None)

def save_report_to_db(report_data, file_metadata, ai_log):
    """
    Save a report and related data to the database in a transaction.
    
    Args:
        report_data: Extracted report data (ReportData object)
        file_metadata: Tuple of (filename, file_size, file_hash)
        ai_log: AI extraction log data (AIExtractionLog object)
        
    Returns:
        Report: The saved Report object
    """
    try:
        # Start a transaction
        filename, file_size, file_hash = file_metadata
        
        # Create a new Report object
        # Handle both attribute access (Pydantic) and dictionary access patterns
        is_dict = isinstance(report_data, dict)
        
        report = Report(
            report_title=report_data['report_title'] if is_dict else report_data.report_title,
            audit_organization=report_data['audit_organization'] if is_dict else report_data.audit_organization,
            publication_year=report_data['publication_year'] if is_dict else report_data.publication_year,
            publication_month=report_data['publication_month'] if is_dict else report_data.publication_month,
            publication_day=report_data['publication_day'] if is_dict else report_data.publication_day,
            overall_conclusion=report_data['overall_conclusion'] if is_dict else report_data.overall_conclusion,
            llm_insight=report_data['llm_insight'] if is_dict else report_data.llm_insight,
            potential_objective_summary=report_data['potential_objective_summary'] if is_dict else report_data.potential_objective_summary,
            original_report_source_url=report_data['original_report_source_url'] if is_dict else report_data.original_report_source_url,
            state=report_data['state'] if is_dict else report_data.state,
            audit_scope=report_data['audit_scope'] if is_dict else report_data.audit_scope,
            original_filename=filename,
            file_hash=file_hash,
            file_size_bytes=file_size,
            status='published',
            processing_status='extracted'
        )
        
        # Add the report to the session
        db.session.add(report)
        db.session.flush()  # Flush to get the report ID
        
        # Add objectives
        if is_dict:
            for obj in report_data['objectives']:
                objective = Objective(
                    report_id=report.id,
                    objective_text=obj['objective_text']
                )
                db.session.add(objective)
        else:
            for obj in report_data.objectives:
                objective = Objective(
                    report_id=report.id,
                    objective_text=obj.objective_text
                )
                db.session.add(objective)
        
        # Add findings
        if is_dict:
            for idx, f in enumerate(report_data['findings']):
                finding = Finding(
                    report_id=report.id,
                    finding_text=f['finding_text'],
                    financial_impact=f.get('financial_impact')
                )
                db.session.add(finding)
        else:
            for idx, f in enumerate(report_data.findings):
                finding = Finding(
                    report_id=report.id,
                    finding_text=f.finding_text,
                    financial_impact=f.financial_impact
                )
                db.session.add(finding)
        
        # Add recommendations
        if is_dict:
            for rec in report_data['recommendations']:
                recommendation = Recommendation(
                    report_id=report.id,
                    recommendation_text=rec['recommendation_text']
                )
                db.session.add(recommendation)
        else:
            for rec in report_data.recommendations:
                recommendation = Recommendation(
                    report_id=report.id,
                    recommendation_text=rec.recommendation_text
                )
                db.session.add(recommendation)
        
        # Add keywords
        if is_dict:
            for kw in report_data['extracted_keywords']:
                keyword = Keyword(
                    report_id=report.id,
                    keyword_text=kw
                )
                db.session.add(keyword)
        else:
            for kw in report_data.extracted_keywords:
                keyword = Keyword(
                    report_id=report.id,
                    keyword_text=kw
                )
                db.session.add(keyword)
        
        # Add AI processing log
        is_ai_log_dict = isinstance(ai_log, dict)
        
        ai_processing_log = AIProcessingLog(
            report_id=report.id,
            model_name=ai_log['model_name'] if is_ai_log_dict else ai_log.model_name,
            input_tokens=ai_log['input_tokens'] if is_ai_log_dict else ai_log.input_tokens,
            output_tokens=ai_log['output_tokens'] if is_ai_log_dict else ai_log.output_tokens,
            total_tokens=ai_log['total_tokens'] if is_ai_log_dict else ai_log.total_tokens,
            input_cost=ai_log['input_cost'] if is_ai_log_dict else ai_log.input_cost,
            output_cost=ai_log['output_cost'] if is_ai_log_dict else ai_log.output_cost,
            total_cost=ai_log['total_cost'] if is_ai_log_dict else ai_log.total_cost,
            processing_time_ms=ai_log['processing_time_ms'] if is_ai_log_dict else ai_log.processing_time_ms,
            extraction_status=ai_log['extraction_status'] if is_ai_log_dict else ai_log.extraction_status,
            error_details=ai_log['error_details'] if is_ai_log_dict and 'error_details' in ai_log else (ai_log.error_details if not is_ai_log_dict else None)
        )
        db.session.add(ai_processing_log)
        
        # Commit the transaction
        db.session.commit()
        
        return report
    
    except Exception as e:
        # Rollback the transaction in case of error
        db.session.rollback()
        logging.error(f"Error saving report to database: {e}")
        raise ValueError(f"Failed to save report to database: {e}")

def update_report_in_db(report_id, updated_data):
    """
    Update an existing report with edited data.
    
    Args:
        report_id: ID of the report to update
        updated_data: Dictionary containing updated report data
        
    Returns:
        Report: The updated Report object
    """
    try:
        # Start a transaction
        report = Report.query.get(report_id)
        if not report:
            raise ValueError(f"Report with ID {report_id} not found")
        
        # Update main report fields
        for key, value in updated_data.get('report', {}).items():
            if hasattr(report, key):
                setattr(report, key, value)
        
        # Update objectives
        if 'objectives' in updated_data:
            # Delete existing objectives
            Objective.query.filter_by(report_id=report_id).delete()
            
            # Add updated objectives
            for obj_data in updated_data['objectives']:
                objective = Objective(
                    report_id=report_id,
                    objective_text=obj_data['objective_text']
                )
                db.session.add(objective)
        
        # Update findings
        if 'findings' in updated_data:
            # Delete existing findings
            Finding.query.filter_by(report_id=report_id).delete()
            
            # Add updated findings
            for find_data in updated_data['findings']:
                finding = Finding(
                    report_id=report_id,
                    finding_text=find_data['finding_text'],
                    financial_impact=find_data.get('financial_impact')
                )
                db.session.add(finding)
        
        # Update recommendations
        if 'recommendations' in updated_data:
            # Delete existing recommendations
            Recommendation.query.filter_by(report_id=report_id).delete()
            
            # Add updated recommendations
            for rec_data in updated_data['recommendations']:
                recommendation = Recommendation(
                    report_id=report_id,
                    recommendation_text=rec_data['recommendation_text']
                )
                db.session.add(recommendation)
        
        # Update keywords
        if 'keywords' in updated_data:
            # Delete existing keywords
            Keyword.query.filter_by(report_id=report_id).delete()
            
            # Add updated keywords
            for kw_data in updated_data['keywords']:
                keyword = Keyword(
                    report_id=report_id,
                    keyword_text=kw_data
                )
                db.session.add(keyword)
        
        # Update timestamp
        report.updated_at = datetime.utcnow()
        
        # Commit the transaction
        db.session.commit()
        
        return report
    
    except Exception as e:
        # Rollback the transaction in case of error
        db.session.rollback()
        logging.error(f"Error updating report in database: {e}")
        raise ValueError(f"Failed to update report in database: {e}")
