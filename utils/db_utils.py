import logging
import json
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
            pdf_storage_path="",  # Empty string as we no longer store PDFs
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
        
        # Add keywords - with many-to-many relationship
        if is_dict:
            keyword_texts = report_data['extracted_keywords']
        else:
            keyword_texts = report_data.extracted_keywords
            
        for kw_text in keyword_texts:
            # Check if keyword already exists
            keyword = Keyword.query.filter_by(keyword_text=kw_text).first()
            if not keyword:
                # Create new keyword if it doesn't exist
                keyword = Keyword(keyword_text=kw_text)
                db.session.add(keyword)
                db.session.flush()  # Flush to get the keyword ID
            
            # Add keyword to report's keywords collection
            report.keywords.append(keyword)
        
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
        
        # Update keywords - with many-to-many relationship
        if 'keywords' in updated_data:
            # Clear existing keyword associations
            report.keywords = []
            
            # Add updated keywords
            for kw_text in updated_data['keywords']:
                # Check if keyword already exists
                keyword = Keyword.query.filter_by(keyword_text=kw_text).first()
                if not keyword:
                    # Create new keyword if it doesn't exist
                    keyword = Keyword(keyword_text=kw_text)
                    db.session.add(keyword)
                    db.session.flush()  # Flush to get the keyword ID
                
                # Add keyword to report's keywords collection
                report.keywords.append(keyword)
        
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


def print_report_data(report_id=None, report=None):
    """
    Print the report data in a structured format to the console.
    
    Args:
        report_id: ID of the report to print (optional if report object is provided)
        report: Report object to print (optional if report_id is provided)
        
    Returns:
        None
    """
    try:
        # Get the report if only ID is provided
        if report is None and report_id is not None:
            report = Report.query.get(report_id)
            
        if report is None:
            logging.error("No report provided for printing")
            return
            
        # Create a structured representation
        report_dict = {
            "REPORT DETAILS": {
                "ID": report.id,
                "Title": report.report_title,
                "Audit Organization": report.audit_organization,
                "Publication Date": f"{report.publication_year}-{report.publication_month:02d}-{report.publication_day or 'N/A'}",
                "State": report.state,
                "Audit Scope": report.audit_scope,
                "File Details": {
                    "Original Filename": report.original_filename,
                    "File Size": f"{report.file_size_bytes:,} bytes",
                    "File Hash": report.file_hash
                },
                "Overall Conclusion": report.overall_conclusion,
                "AI Insight": report.llm_insight,
                "Potential Objective Summary": report.potential_objective_summary,
                "Original Report URL": report.original_report_source_url or "N/A",
                "Status": report.status,
                "Processing Status": report.processing_status,
                "Created": report.created_at.strftime("%Y-%m-%d %H:%M:%S") if report.created_at else "N/A",
                "Last Updated": report.updated_at.strftime("%Y-%m-%d %H:%M:%S") if report.updated_at else "N/A"
            },
            "OBJECTIVES": [],
            "FINDINGS": [],
            "RECOMMENDATIONS": [],
            "KEYWORDS": [],
            "AI PROCESSING LOGS": []
        }
        
        # Add objectives
        for obj in report.objectives:
            report_dict["OBJECTIVES"].append({
                "ID": obj.id,
                "Text": obj.objective_text
            })
            
        # Add findings
        for finding in report.findings:
            report_dict["FINDINGS"].append({
                "ID": finding.id,
                "Text": finding.finding_text,
                "Financial Impact": f"${finding.financial_impact:,.2f}" if finding.financial_impact else "N/A"
            })
            
        # Add recommendations
        for rec in report.recommendations:
            report_dict["RECOMMENDATIONS"].append({
                "ID": rec.id,
                "Text": rec.recommendation_text,
                "Related Finding ID": rec.related_finding_id or "N/A"
            })
            
        # Add keywords
        for kw in report.keywords:
            report_dict["KEYWORDS"].append(kw.keyword_text)
            
        # Add AI processing logs
        for log in report.ai_logs:
            report_dict["AI PROCESSING LOGS"].append({
                "ID": log.id,
                "Model": log.model_name,
                "Status": log.extraction_status,
                "Tokens": f"{log.total_tokens:,} ({log.input_tokens:,} in, {log.output_tokens:,} out)",
                "Cost": f"${log.total_cost:.6f}",
                "Processing Time": f"{log.processing_time_ms:,} ms",
                "Error": log.error_details or "None"
            })
        
        # Print the structured report
        print("\n" + "="*80)
        print(f"REPORT DATA FOR: {report.report_title}")
        print("="*80 + "\n")
        
        # Print the report in a nicely formatted way
        print(json.dumps(report_dict, indent=2))
        
        print("\n" + "="*80)
        print("END OF REPORT DATA")
        print("="*80 + "\n")
        
    except Exception as e:
        logging.error(f"Error printing report data: {e}")
        print(f"Error printing report data: {e}")
