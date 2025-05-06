import os
import io
import json
import hashlib
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from app import db
from models import Report, Finding, Recommendation, Objective, Keyword, AIProcessingLog
from utils.pdf_utils import (
    extract_text_from_pdf, extract_text_from_pdf_memory,
    extract_keywords_from_pdf_metadata, extract_keywords_from_pdf_metadata_memory,
    process_keywords
)
from utils.ai_extraction import extract_data_with_openai
from utils.db_utils import check_duplicate_report, save_report_to_db, update_report_in_db

def register_routes(app):
    @app.route('/')
    def dashboard():
        """Admin dashboard landing page"""
        # Get some stats for the dashboard
        total_reports = Report.query.count()
        recent_reports = Report.query.order_by(Report.created_at.desc()).limit(5).all()
        featured_reports = Report.query.filter_by(featured=True).all()
        
        return render_template('dashboard.html', 
                            total_reports=total_reports,
                            recent_reports=recent_reports,
                            featured_reports=featured_reports)
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        """Page for uploading PDF reports"""
        if request.method == 'POST':
            # Check if files were uploaded
            if 'files' not in request.files:
                flash('No files selected', 'danger')
                return redirect(request.url)
            
            files = request.files.getlist('files')
            
            # If no files were selected
            if len(files) == 0 or files[0].filename == '':
                flash('No files selected', 'danger')
                return redirect(request.url)
            
            # Select the AI model
            ai_model = request.form.get('ai_model', 'openai')
            
            upload_results = []
            
            for file in files:
                if file.filename == '':
                    continue
                
                # Check if the file is a PDF
                if not file.filename.lower().endswith('.pdf'):
                    upload_results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'message': 'Only PDF files are allowed'
                    })
                    continue
                
                try:
                    # Process the file in memory without saving to disk
                    file_content = file.read()
                    file_size = len(file_content)
                    filename = secure_filename(file.filename)
                    
                    # Calculate file hash from memory
                    file_hash = hashlib.sha256(file_content).hexdigest()
                    
                    # Check for duplicates
                    is_duplicate, existing_report, reason = check_duplicate_report(file_hash, filename)
                    
                    if is_duplicate:
                        # Handle based on duplication reason
                        if reason == 'file_hash':
                            upload_results.append({
                                'filename': file.filename,
                                'status': 'duplicate',
                                'message': f'Report with same content already exists (ID: {existing_report.id})',
                                'report_id': existing_report.id
                            })
                        else:  # filename match
                            upload_results.append({
                                'filename': file.filename,
                                'status': 'warning',
                                'message': f'Report with same filename already exists (ID: {existing_report.id}). Content is different.',
                                'report_id': existing_report.id
                            })
                        continue
                    
                    # Create a BytesIO object from file content
                    pdf_io = io.BytesIO(file_content)
                    
                    # Extract text from PDF in memory
                    pdf_text = extract_text_from_pdf_memory(pdf_io)
                    
                    # Extract keywords from PDF metadata in memory
                    pdf_metadata_keywords = extract_keywords_from_pdf_metadata_memory(pdf_io)
                    
                    # Use AI model to extract data
                    if ai_model == 'openai':
                        api_key = app.config.get('OPENAI_API_KEY')
                        if not api_key:
                            raise ValueError("OpenAI API key not configured")
                        
                        # Extract data and get AI log
                        report_data, ai_log = extract_data_with_openai(pdf_text, api_key)
                        
                        # Combine and deduplicate keywords from PDF metadata and AI extraction
                        combined_keywords = process_keywords(pdf_metadata_keywords, report_data.extracted_keywords)
                        
                        # Update the report data with combined keywords
                        report_data_dict = report_data.dict()
                        report_data_dict['extracted_keywords'] = combined_keywords
                        
                        # Create file metadata tuple for in-memory processing
                        # (filename, file_size, file_hash) - no file path needed
                        file_metadata = (filename, file_size, file_hash)
                        
                        # Redirect to review page with extraction ID
                        upload_results.append({
                            'filename': file.filename,
                            'status': 'success',
                            'message': 'Processing completed successfully',
                            'temp_id': file_hash,  # Use file hash as a temporary ID for now
                            'report_data': report_data_dict,
                            'ai_log': ai_log.dict(),
                            'pdf_metadata_keywords': pdf_metadata_keywords,
                            'file_metadata': file_metadata
                        })
                        
                    else:
                        upload_results.append({
                            'filename': file.filename,
                            'status': 'error',
                            'message': f'Unsupported AI model: {ai_model}'
                        })
                
                except Exception as e:
                    logging.error(f"Error processing file {file.filename}: {e}")
                    upload_results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'message': str(e)
                    })
            
            # Store processing results in session for review
            if any(result['status'] == 'success' for result in upload_results):
                # Save successful extractions to session for review
                success_results = [result for result in upload_results if result['status'] == 'success']
                
                if len(success_results) == 1:
                    # If only one file was successfully processed, redirect to review page
                    result = success_results[0]
                    # Store data in session
                    temp_id = result['temp_id']
                    app.config[f'temp_extraction_{temp_id}'] = {
                        'report_data': result['report_data'],
                        'ai_log': result['ai_log'],
                        'file_metadata': result['file_metadata']
                    }
                    return redirect(url_for('review', temp_id=temp_id))
                else:
                    # If multiple files were processed, show summary
                    flash(f'Processed {len(success_results)} files successfully', 'success')
                    return render_template('upload.html', results=upload_results)
            
            # If no files were processed successfully
            flash('No files were processed successfully', 'warning')
            return render_template('upload.html', results=upload_results)
        
        # GET request - show upload form
        return render_template('upload.html')
    
    @app.route('/review/<temp_id>', methods=['GET', 'POST'])
    def review(temp_id):
        """Page for reviewing and editing extracted data"""
        # Get extraction data from session
        extraction_data = app.config.get(f'temp_extraction_{temp_id}')
        
        if not extraction_data:
            flash('Extraction data not found or expired', 'danger')
            return redirect(url_for('upload'))
        
        report_data = extraction_data['report_data']
        ai_log = extraction_data['ai_log']
        file_metadata = extraction_data['file_metadata']
        
        if request.method == 'POST':
            try:
                # Get updated data from form
                updated_data = request.form.get('report_data')
                if not updated_data:
                    raise ValueError("No report data provided")
                
                # Parse JSON data
                updated_data = json.loads(updated_data)
                
                # Save to database
                report = save_report_to_db(
                    updated_data,
                    file_metadata,
                    ai_log
                )
                
                # Clean up session data
                app.config.pop(f'temp_extraction_{temp_id}', None)
                
                flash('Report saved successfully', 'success')
                return redirect(url_for('reports'))
            
            except Exception as e:
                logging.error(f"Error saving report: {e}")
                flash(f'Error saving report: {str(e)}', 'danger')
                return render_template('review.html', 
                                    report_data=report_data,
                                    ai_log=ai_log,
                                    temp_id=temp_id)
        
        # GET request - show review form
        return render_template('review.html', 
                            report_data=report_data,
                            ai_log=ai_log,
                            temp_id=temp_id,
                            pdf_path=file_metadata[1])
    
    @app.route('/reports')
    def reports():
        """Page for viewing all reports"""
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        reports = Report.query.order_by(Report.created_at.desc()).paginate(page=page, per_page=per_page)
        
        return render_template('reports.html', reports=reports)
    
    @app.route('/report/<int:report_id>')
    def report_detail(report_id):
        """Page for viewing a single report's details"""
        report = Report.query.get_or_404(report_id)
        
        return render_template('report_detail.html', report=report)
    
    @app.route('/report/<int:report_id>/edit', methods=['GET', 'POST'])
    def report_edit(report_id):
        """Page for editing a report"""
        report = Report.query.get_or_404(report_id)
        
        if request.method == 'POST':
            try:
                # Get updated data from form
                updated_data = request.form.get('report_data')
                if not updated_data:
                    raise ValueError("No report data provided")
                
                # Parse JSON data
                updated_data = json.loads(updated_data)
                
                # Update in database
                report = update_report_in_db(report_id, updated_data)
                
                flash('Report updated successfully', 'success')
                return redirect(url_for('report_detail', report_id=report.id))
            
            except Exception as e:
                logging.error(f"Error updating report: {e}")
                flash(f'Error updating report: {str(e)}', 'danger')
        
        # Prepare report data for the template
        report_data = {
            'report': {
                'id': report.id,
                'report_title': report.report_title,
                'audit_organization': report.audit_organization,
                'publication_year': report.publication_year,
                'publication_month': report.publication_month,
                'publication_day': report.publication_day,
                'overall_conclusion': report.overall_conclusion,
                'llm_insight': report.llm_insight,
                'potential_objective_summary': report.potential_objective_summary,
                'original_report_source_url': report.original_report_source_url,
                'state': report.state,
                'audit_scope': report.audit_scope,
            },
            'objectives': [{'objective_text': obj.objective_text} for obj in report.objectives],
            'findings': [{'finding_text': f.finding_text, 'financial_impact': f.financial_impact} for f in report.findings],
            'recommendations': [{'recommendation_text': r.recommendation_text} for r in report.recommendations],
            'keywords': [kw.keyword_text for kw in report.keywords]
        }
        
        return render_template('report_edit.html', 
                            report=report,
                            report_data=report_data)
    
    @app.route('/report/<int:report_id>/toggle_featured', methods=['POST'])
    def toggle_featured(report_id):
        """Toggle featured status of a report"""
        report = Report.query.get_or_404(report_id)
        
        try:
            report.featured = not report.featured
            db.session.commit()
            
            return jsonify({
                'success': True,
                'featured': report.featured
            })
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error toggling featured status: {e}")
            
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/uploads/<path:filename>')
    def serve_pdf(filename):
        """Serve uploaded PDF files"""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
