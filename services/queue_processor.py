# services/queue_processor.py
import requests
import hashlib
import io
from datetime import datetime

from models import ScrapingQueue, Report, DuplicateCheck
from app import db, app
from utils.pdf_utils import extract_text_from_pdf_memory, get_file_hash_memory
from utils.ai_extraction import extract_data_with_ai


class QueueProcessor:
    def __init__(self):
        pass
        
    def process_queue(self):
        """Process all pending items in queue."""
        with app.app_context():
            while True:
                item = self._get_next_item()
                if not item:
                    break
                    
                self._process_item(item)
    
    def _get_next_item(self):
        """Get next pending item from queue."""
        return db.session.query(ScrapingQueue).filter_by(
            status='pending'
        ).filter(
            ScrapingQueue.retry_count < 3  # Max 3 attempts
        ).order_by(
            ScrapingQueue.created_at
        ).first()
    
    def _process_item(self, item):
        """Process a single queue item."""
        try:
            # Check if this is an uploaded file or a URL to download
            is_upload = item.source_domain == "manual_upload"
            
            if is_upload:
                # Update status for uploaded file
                item.status = 'processing'
                db.session.commit()
                
                # Extract file content from metadata
                file_content_hex = item.document_metadata.get('file_content')
                if not file_content_hex:
                    raise ValueError("No file content found in uploaded item metadata")
                
                # Convert hex string back to bytes
                pdf_content = bytes.fromhex(file_content_hex)
                file_hash = item.document_metadata.get('file_hash')
                
            else:
                # Update status for URL download
                item.status = 'downloading'
                db.session.commit()
                
                # Download PDF from URL
                response = requests.get(item.url, timeout=30)
                response.raise_for_status()
                
                pdf_content = response.content
                # Calculate hash
                file_hash = hashlib.sha256(pdf_content).hexdigest()
            
            # Check if duplicate by hash
            existing = db.session.query(Report).filter_by(
                file_hash=file_hash
            ).first()
            
            if existing:
                item.status = 'duplicate'
                item.report_id = existing.id
                # Record duplicate
                dup_check = DuplicateCheck(
                    queue_item_id=item.id,
                    existing_report_id=existing.id
                )
                db.session.add(dup_check)
            else:
                # Process through existing pipeline
                # Get AI provider preference from item metadata or default to openai
                ai_provider = item.document_metadata.get('ai_provider', 'openai') if item.document_metadata else 'openai'
                report = self._create_report(item, pdf_content, is_upload, ai_provider)
                if report:
                    item.status = 'completed'
                    item.report_id = report.id
                else:
                    item.status = 'failed'
                    item.error_message = "Failed to process PDF content"
                
        except Exception as e:
            item.status = 'failed'
            item.error_message = str(e)
            item.retry_count += 1
            
            # Reset to pending if retries remaining
            if item.retry_count < 3:
                item.status = 'pending'
        
        finally:
            item.completed_at = datetime.utcnow()
            db.session.commit()
    
    def _create_report(self, queue_item, pdf_content, is_upload=False, ai_provider="openai"):
        """Create a new report from PDF content (downloaded or uploaded)."""
        try:
            # Create BytesIO object for PDF processing
            pdf_io = io.BytesIO(pdf_content)
            
            # Extract text
            extracted_text = extract_text_from_pdf_memory(pdf_io)
            
            # Calculate file hash
            file_hash = hashlib.sha256(pdf_content).hexdigest()
            
            # Extract data with AI using specified provider
            report_data, ai_log = extract_data_with_ai(extracted_text, provider=ai_provider)
            
            # Create report object
            # Get the appropriate filename and URL based on source
            if is_upload:
                original_filename = queue_item.document_metadata.get('original_filename', queue_item.title)
                source_url = f"Manual Upload: {original_filename}"
            else:
                original_filename = queue_item.title + '.pdf'
                source_url = queue_item.url
            
            report = Report(
                report_title=report_data.report_title,
                audit_organization=report_data.audit_organization,
                publication_year=report_data.publication_year,
                publication_month=report_data.publication_month,
                publication_day=report_data.publication_day,
                overall_conclusion=report_data.overall_conclusion,
                llm_insight=report_data.llm_insight,
                potential_objective_summary=report_data.potential_objective_summary,
                original_report_source_url=source_url,
                state=report_data.state,
                audit_scope=report_data.audit_scope,
                original_filename=original_filename,
                file_hash=file_hash,
                file_size_bytes=len(pdf_content),
                status='completed',
                processing_status='completed'
            )
            
            db.session.add(report)
            db.session.flush()  # Get the ID
            
            # Add extracted entities
            from models import Finding, Recommendation, Objective, Keyword, AIProcessingLog
            
            # Add findings
            for finding_text in report_data.findings:
                finding = Finding(
                    report_id=report.id,
                    finding_text=finding_text
                )
                db.session.add(finding)
            
            # Add recommendations
            for rec_text in report_data.recommendations:
                recommendation = Recommendation(
                    report_id=report.id,
                    recommendation_text=rec_text
                )
                db.session.add(recommendation)
            
            # Add objectives
            for obj_text in report_data.objectives:
                objective = Objective(
                    report_id=report.id,
                    objective_text=obj_text
                )
                db.session.add(objective)
            
            # Add keywords
            for keyword_text in report_data.extracted_keywords:
                # Check if keyword exists
                keyword = db.session.query(Keyword).filter_by(keyword_text=keyword_text).first()
                if not keyword:
                    keyword = Keyword(keyword_text=keyword_text)
                    db.session.add(keyword)
                    db.session.flush()
                
                # Associate with report
                report.keywords.append(keyword)
            
            # Add AI processing log
            ai_processing_log = AIProcessingLog(
                report_id=report.id,
                model_name=ai_log.model_name,
                input_tokens=ai_log.input_tokens,
                output_tokens=ai_log.output_tokens,
                total_tokens=ai_log.total_tokens,
                input_cost=ai_log.input_cost,
                output_cost=ai_log.output_cost,
                total_cost=ai_log.total_cost,
                processing_time_ms=ai_log.processing_time_ms,
                extraction_status=ai_log.extraction_status,
                error_details=ai_log.error_details
            )
            db.session.add(ai_processing_log)
            
            db.session.commit()
            return report
            
        except Exception as e:
            db.session.rollback()
            raise