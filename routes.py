import os
import io
import json
import hashlib
import logging
import traceback
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app
from werkzeug.utils import secure_filename
from app import db
from models import Report, Finding, Recommendation, Objective, Keyword, AIProcessingLog
from utils.pdf_utils import (
    extract_text_from_pdf_memory,
    extract_keywords_from_pdf_metadata_memory,
    process_keywords, process_uploaded_file_memory
)
from utils.ai_extraction import extract_data_with_openai
from utils.db_utils import check_duplicate_report, save_report_to_db, update_report_in_db, print_report_data
from utils.parser_strategies import ParsingStrategy, get_parser_function
from utils.comparison_storage import ComparisonStorage
from utils.chunking_strategies import ChunkingStrategy, get_chunker_function, calculate_chunk_statistics, count_tokens
from utils.chunking_storage import ChunkingComparisonStorage
from models import ScrapingQueue, SearchHistory, DuplicateCheck
from services.audit_search_service import AuditSearchService
from sqlalchemy import func

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
                            
    @app.route('/parse-review', methods=['GET', 'POST'])
    def parse_review():
        """Page for reviewing raw PDF text extraction"""
        extracted_text = None
        
        if request.method == 'POST' and 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            
            if pdf_file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
                
            if pdf_file:
                try:
                    # Process the uploaded file in memory
                    pdf_io = io.BytesIO(pdf_file.read())
                    
                    # Extract text using PyMuPDF
                    extracted_text = extract_text_from_pdf_memory(pdf_io)
                    
                    # Get file metadata for display
                    filename = secure_filename(pdf_file.filename)
                    file_size = len(pdf_io.getvalue())
                    
                    flash(f'Successfully extracted text from {filename} ({file_size/1024:.1f} KB)', 'success')
                    
                except Exception as e:
                    flash(f'Error extracting text: {str(e)}', 'error')
                    logging.error(f"PDF extraction error: {str(e)}")
        
        return render_template('parse_review.html', extracted_text=extracted_text)
    
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
                    filename, file_size, file_hash, file_content = process_uploaded_file_memory(file)
                    
                    # Check for duplicates in both existing reports and queue
                    is_duplicate, existing_report, reason, is_hidden = check_duplicate_report(file_hash, filename)
                    
                    if is_duplicate:
                        # Handle based on duplication reason
                        if reason == 'file_hash':
                            if is_hidden:
                                upload_results.append({
                                    'filename': file.filename,
                                    'status': 'hidden_duplicate',
                                    'message': f'Report already exists but is hidden (ID: {existing_report.id}). Would you like to restore it?',
                                    'report_id': existing_report.id,
                                    'can_restore': True
                                })
                            else:
                                upload_results.append({
                                    'filename': file.filename,
                                    'status': 'duplicate',
                                    'message': f'Report with same content already exists (ID: {existing_report.id})',
                                    'report_id': existing_report.id
                                })
                        else:  # filename match
                            if is_hidden:
                                upload_results.append({
                                    'filename': file.filename,
                                    'status': 'hidden_warning',
                                    'message': f'Hidden report with same filename exists (ID: {existing_report.id}). Content may be different.',
                                    'report_id': existing_report.id,
                                    'can_restore': True
                                })
                            else:
                                upload_results.append({
                                    'filename': file.filename,
                                    'status': 'warning',
                                    'message': f'Report with same filename already exists (ID: {existing_report.id}). Content is different.',
                                    'report_id': existing_report.id
                                })
                        continue
                    
                    # Check for duplicates in the queue (by URL which we'll use as file hash for uploads)
                    upload_url = f"upload://{file_hash}"  # Create unique identifier for uploads
                    existing_queue_item = ScrapingQueue.query.filter_by(url=upload_url).first()
                    
                    if existing_queue_item:
                        upload_results.append({
                            'filename': file.filename,
                            'status': 'duplicate',
                            'message': f'File already in queue (ID: {existing_queue_item.id})',
                            'queue_id': existing_queue_item.id
                        })
                        continue
                    
                    # Create queue item for uploaded file
                    queue_item = ScrapingQueue(
                        url=upload_url,
                        title=filename,
                        source_domain="manual_upload",
                        document_metadata={
                            'filename': filename,
                            'file_size': file_size,
                            'file_hash': file_hash,
                            'upload_source': 'manual_upload',
                            'original_filename': file.filename,
                            'file_content': file_content.hex()  # Store as hex string
                        },
                        ai_classification={
                            'is_medicaid_audit': True,  # User-selected, assume it's an audit
                            'confidence': 1.0,
                            'source': 'manual_upload',
                            'reasoning': 'File manually uploaded by user'
                        },
                        status='pending_review',  # Goes to review queue
                        user_override=True  # Mark as user-vetted
                    )
                    
                    db.session.add(queue_item)
                    db.session.commit()
                    
                    upload_results.append({
                        'filename': file.filename,
                        'status': 'queued',
                        'message': 'File added to review queue successfully',
                        'queue_id': queue_item.id
                    })
                
                except Exception as e:
                    logging.error(f"Error processing file {file.filename}: {e}")
                    upload_results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'message': str(e)
                    })
            
            # Handle upload results
            queued_count = len([r for r in upload_results if r['status'] == 'queued'])
            duplicate_count = len([r for r in upload_results if r['status'] in ['duplicate', 'hidden_duplicate']])
            warning_count = len([r for r in upload_results if r['status'] in ['warning', 'hidden_warning']])
            error_count = len([r for r in upload_results if r['status'] == 'error'])
            
            if queued_count > 0:
                flash(f'Successfully added {queued_count} file{"s" if queued_count != 1 else ""} to review queue', 'success')
                if duplicate_count > 0:
                    flash(f'{duplicate_count} duplicate file{"s" if duplicate_count != 1 else ""} skipped', 'info')
                if error_count > 0:
                    flash(f'{error_count} file{"s" if error_count != 1 else ""} had errors', 'warning')
                
                # Redirect to review queue to see uploaded files
                return redirect(url_for('queue_review'))
            else:
                # No files queued - show results with errors/duplicates
                if duplicate_count > 0:
                    flash(f'All {duplicate_count} file{"s" if duplicate_count != 1 else ""} were duplicates', 'info')
                if error_count > 0:
                    flash(f'All {error_count} file{"s" if error_count != 1 else ""} had errors', 'danger')
                
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
                
                # Report successfully saved
                
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
                            temp_id=temp_id)
    
    @app.route('/reports')
    def reports():
        """Page for viewing all reports"""
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        # Get sort parameters
        sort_by = request.args.get('sort_by', 'created_at')
        sort_dir = request.args.get('sort_dir', 'desc')
        
        # Only show non-hidden reports
        query = Report.query.filter(Report.hidden == False)
        
        # Apply sorting
        if sort_by == 'title':
            sort_column = Report.report_title
        elif sort_by == 'organization':
            sort_column = Report.audit_organization
        elif sort_by == 'state':
            sort_column = Report.state
        elif sort_by == 'publication_date':
            sort_column = Report.publication_year.desc(), Report.publication_month.desc()
        elif sort_by == 'featured':
            sort_column = Report.featured
        else:  # default to created_at
            sort_column = Report.created_at
        
        if sort_dir == 'asc':
            if sort_by == 'publication_date':
                query = query.order_by(Report.publication_year.asc(), Report.publication_month.asc())
            else:
                query = query.order_by(sort_column.asc())
        else:
            if sort_by == 'publication_date':
                query = query.order_by(Report.publication_year.desc(), Report.publication_month.desc())
            else:
                query = query.order_by(sort_column.desc())
        
        reports = query.paginate(page=page, per_page=per_page)
        
        return render_template('reports.html', reports=reports, sort_by=sort_by, sort_dir=sort_dir)
        
    @app.route('/compare-upload', methods=['GET'])
    def compare_upload():
        """Page for uploading PDFs to compare parsing strategies"""
        parser_choices = ParsingStrategy.choices()
        return render_template('compare_upload.html', parser_choices=parser_choices)
        
    @app.route('/compare-process', methods=['POST'])
    def compare_process():
        """Process uploaded PDF with selected parsers and store results"""
        # Check if files were uploaded
        if 'pdf_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('compare_upload'))
            
        pdf_file = request.files['pdf_file']
        
        # If no file was selected
        if pdf_file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('compare_upload'))
            
        # Check if the file is a PDF
        if not pdf_file.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed', 'danger')
            return redirect(url_for('compare_upload'))
            
        # Get parser selections
        parser_key_1 = request.form.get('parser_key_1')
        parser_key_2 = request.form.get('parser_key_2')
        
        # Check if parsers are selected
        if not parser_key_1 or not parser_key_2:
            flash('Please select two parsing strategies', 'danger')
            return redirect(url_for('compare_upload'))
        
        # Check if AI extraction is enabled
        run_ai_extraction = request.form.get('run_ai_extraction') == 'on'
        
        try:
            # Initialize response data
            comparison_data = {
                'filename': secure_filename(pdf_file.filename),
                'parser_key_1': parser_key_1,
                'parser_key_2': parser_key_2,
                'parser_name_1': ParsingStrategy[parser_key_1].value if parser_key_1 in ParsingStrategy.__members__ else parser_key_1,
                'parser_name_2': ParsingStrategy[parser_key_2].value if parser_key_2 in ParsingStrategy.__members__ else parser_key_2,
                'raw_text_1': None,
                'raw_text_2': None,
                'structured_data_1': None,
                'structured_data_2': None,
                'ai_log_1': None,
                'ai_log_2': None,
                'error_1': None,
                'error_2': None,
                'run_ai_extraction': run_ai_extraction
            }
            
            # Process the file in memory
            pdf_content = pdf_file.read()
            
            # Create BytesIO object for the PDF content
            pdf_io_1 = io.BytesIO(pdf_content)
            pdf_io_2 = io.BytesIO(pdf_content)
            
            # Process with parser 1
            try:
                parser_func_1 = get_parser_function(parser_key_1)
                raw_text_1 = parser_func_1(pdf_io_1)
                comparison_data['raw_text_1'] = raw_text_1
                
                # If AI extraction is enabled, run it
                if run_ai_extraction:
                    try:
                        api_key = app.config.get('OPENAI_API_KEY')
                        if not api_key:
                            comparison_data['error_1'] = "OpenAI API key not configured"
                        else:
                            report_data_1, ai_log_1 = extract_data_with_openai(raw_text_1, api_key)
                            comparison_data['structured_data_1'] = report_data_1.dict()
                            comparison_data['ai_log_1'] = ai_log_1.dict()
                    except Exception as e:
                        logging.error(f"Error in AI extraction for parser 1: {e}")
                        comparison_data['error_1'] = f"AI extraction error: {str(e)}"
            except Exception as e:
                logging.error(f"Error in parser 1 ({parser_key_1}): {e}")
                comparison_data['error_1'] = str(e)
            
            # Process with parser 2
            try:
                parser_func_2 = get_parser_function(parser_key_2)
                raw_text_2 = parser_func_2(pdf_io_2)
                comparison_data['raw_text_2'] = raw_text_2
                
                # If AI extraction is enabled, run it
                if run_ai_extraction:
                    try:
                        api_key = app.config.get('OPENAI_API_KEY')
                        if not api_key:
                            comparison_data['error_2'] = "OpenAI API key not configured"
                        else:
                            report_data_2, ai_log_2 = extract_data_with_openai(raw_text_2, api_key)
                            comparison_data['structured_data_2'] = report_data_2.dict()
                            comparison_data['ai_log_2'] = ai_log_2.dict()
                    except Exception as e:
                        logging.error(f"Error in AI extraction for parser 2: {e}")
                        comparison_data['error_2'] = f"AI extraction error: {str(e)}"
            except Exception as e:
                logging.error(f"Error in parser 2 ({parser_key_2}): {e}")
                comparison_data['error_2'] = str(e)
            
            # Store comparison data
            storage = ComparisonStorage(app)
            comparison_id = storage.store_comparison(comparison_data)
            
            # Redirect to comparison review page
            return redirect(url_for('compare_review', comparison_id=comparison_id))
            
        except Exception as e:
            logging.error(f"Error processing PDF for comparison: {e}")
            logging.error(traceback.format_exc())
            flash(f'Error processing PDF: {str(e)}', 'danger')
            return redirect(url_for('compare_upload'))
            
    @app.route('/compare-review/<comparison_id>')
    def compare_review(comparison_id):
        """Page for reviewing parser comparison results"""
        # Access stored comparison data
        storage = ComparisonStorage(app)
        comparison_data = storage.get_comparison(comparison_id)
        
        if not comparison_data:
            flash('Comparison data not found or expired', 'danger')
            return redirect(url_for('compare_upload'))
            
        return render_template('compare_review.html', 
                             comparison_id=comparison_id,
                             comparison_data=comparison_data)
                             
    @app.route('/api/comparison/<comparison_id>')
    def get_comparison(comparison_id):
        """API endpoint for fetching comparison data"""
        storage = ComparisonStorage(app)
        comparison_data = storage.get_comparison(comparison_id)
        
        if not comparison_data:
            return jsonify({'error': 'Comparison data not found or expired'}), 404
            
        # We need to limit the size of the response
        # Let's create a copy with truncated raw text if it's too large
        response_data = dict(comparison_data)
        
        if response_data.get('raw_text_1') and len(response_data['raw_text_1']) > 100000:
            response_data['raw_text_1'] = response_data['raw_text_1'][:100000] + "\n\n... [truncated] ..."
            response_data['raw_text_1_truncated'] = True
        
        if response_data.get('raw_text_2') and len(response_data['raw_text_2']) > 100000:
            response_data['raw_text_2'] = response_data['raw_text_2'][:100000] + "\n\n... [truncated] ..."
            response_data['raw_text_2_truncated'] = True
            
        return jsonify(response_data)
    
    @app.route('/report/<int:report_id>')
    def report_detail(report_id):
        """Page for viewing a single report's details"""
        report = Report.query.get_or_404(report_id)
        
        # Report detail access
        
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
                
                # Report successfully updated
                
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
    
    @app.route('/report/<int:report_id>/hide', methods=['POST'])
    def hide_report(report_id):
        """Hide a report (soft delete)"""
        report = Report.query.get_or_404(report_id)
        
        try:
            report.hidden = True
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Report "{report.report_title}" has been hidden'
            })
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error hiding report: {e}")
            
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/report/<int:report_id>/unhide', methods=['POST'])
    def unhide_report(report_id):
        """Unhide a report (restore from soft delete)"""
        report = Report.query.filter_by(id=report_id, hidden=True).first_or_404()
        
        try:
            report.hidden = False
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Report "{report.report_title}" has been restored'
            })
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error unhiding report: {e}")
            
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # PDF serving endpoint removed as we no longer store PDFs on disk
    
    @app.route('/compare-chunks/<report_id>')
    def compare_chunks(report_id):
        """Page for selecting chunking strategies to compare"""
        report = Report.query.get_or_404(report_id)
        
        # Get the options for the chunking strategies
        chunking_strategies = ChunkingStrategy.choices()
        
        return render_template('compare_chunks.html', 
                              report=report,
                              chunking_strategies=chunking_strategies)
                              
    @app.route('/process-chunks', methods=['POST'])
    def process_chunks():
        """Process a report with selected chunking strategies"""
        report_id = request.form.get('report_id')
        strategy_1 = request.form.get('strategy_1')
        strategy_2 = request.form.get('strategy_2')
        
        # Get strategy parameters from form
        # Convert form data with proper typing (ints, bools, etc.)
        params_1 = {}
        params_2 = {}
        
        # Extract and convert parameters for strategy 1
        for key in request.form:
            if key.startswith('params_1_'):
                param_name = key[9:]  # Remove 'params_1_' prefix
                value = request.form[key]
                
                # Handle different parameter types
                if value.isdigit():
                    params_1[param_name] = int(value)
                elif value.lower() in ('true', 'false'):
                    params_1[param_name] = value.lower() == 'true'
                else:
                    params_1[param_name] = value
        
        # Extract and convert parameters for strategy 2
        for key in request.form:
            if key.startswith('params_2_'):
                param_name = key[9:]  # Remove 'params_2_' prefix
                value = request.form[key]
                
                # Handle different parameter types
                if value.isdigit():
                    params_2[param_name] = int(value)
                elif value.lower() in ('true', 'false'):
                    params_2[param_name] = value.lower() == 'true'
                else:
                    params_2[param_name] = value
                    
        # Validate report ID
        if not report_id:
            flash('Report ID is required', 'danger')
            return redirect(url_for('dashboard'))
            
        # Validate strategies
        if not strategy_1 or not strategy_2:
            flash('Two chunking strategies must be selected', 'danger')
            return redirect(url_for('compare_chunks', report_id=report_id))
            
        try:
            # Get report text
            report = Report.query.get_or_404(report_id)
            
            # Get the clean text from the report
            # Since we don't save the PDF content, we'll use the combined findings, recommendations,
            # objectives, and overall conclusion as the text to chunk
            report_text = f"{report.report_title}\n\n"
            
            if report.overall_conclusion:
                report_text += f"OVERALL CONCLUSION\n{report.overall_conclusion}\n\n"
                
            if report.objectives:
                report_text += "OBJECTIVES\n"
                for obj in report.objectives:
                    report_text += f"- {obj.objective_text}\n"
                report_text += "\n"
                
            if report.findings:
                report_text += "FINDINGS\n"
                for finding in report.findings:
                    report_text += f"- {finding.finding_text}\n"
                report_text += "\n"
                
            if report.recommendations:
                report_text += "RECOMMENDATIONS\n"
                for rec in report.recommendations:
                    report_text += f"- {rec.recommendation_text}\n"
                report_text += "\n"
                
            # Add the report insight if available
            if report.llm_insight:
                report_text += f"AI INSIGHT\n{report.llm_insight}\n\n"
            
            # Get chunking functions
            chunker_1 = get_chunker_function(strategy_1)
            chunker_2 = get_chunker_function(strategy_2)
            
            # Get parameter models for each strategy
            strategy_1_enum = ChunkingStrategy[strategy_1]
            strategy_2_enum = ChunkingStrategy[strategy_2]
            
            # Create parameter models
            params_model_1 = strategy_1_enum.param_model(**params_1)
            params_model_2 = strategy_2_enum.param_model(**params_2)
            
            # Generate chunks
            chunks_1 = chunker_1(report_text, params_model_1)
            chunks_2 = chunker_2(report_text, params_model_2)
            
            # Calculate statistics
            stats_1 = calculate_chunk_statistics(chunks_1)
            stats_2 = calculate_chunk_statistics(chunks_2)
            
            # Create comparison data
            comparison_data = {
                'report_id': report_id,
                'report_title': report.report_title,
                'strategy_1': {
                    'name': strategy_1,
                    'display_name': strategy_1_enum.display_name,
                    'params': params_model_1.dict()
                },
                'strategy_2': {
                    'name': strategy_2,
                    'display_name': strategy_2_enum.display_name,
                    'params': params_model_2.dict()
                },
                'chunks_1': chunks_1,
                'chunks_2': chunks_2,
                'stats_1': stats_1,
                'stats_2': stats_2
            }
            
            # Store comparison data
            storage = ChunkingComparisonStorage(app)
            comparison_id = storage.store_chunking_comparison(comparison_data)
            
            # Redirect to comparison review page
            return redirect(url_for('chunks_review', comparison_id=comparison_id))
            
        except Exception as e:
            logging.error(f"Error processing chunks: {traceback.format_exc()}")
            flash(f'Error processing chunks: {str(e)}', 'danger')
            return redirect(url_for('compare_chunks', report_id=report_id))
            
    @app.route('/chunks-review/<comparison_id>')
    def chunks_review(comparison_id):
        """Page for reviewing chunking comparison results"""
        storage = ChunkingComparisonStorage(app)
        comparison_data = storage.get_chunking_comparison(comparison_id)
        
        if not comparison_data:
            flash('Chunking comparison data not found or expired', 'danger')
            return redirect(url_for('dashboard'))
            
        return render_template('chunks_review.html', 
                               comparison_id=comparison_id,
                               comparison_data=comparison_data)
                               
    @app.route('/api/chunk-comparison/<comparison_id>')
    def get_chunk_comparison(comparison_id):
        """API endpoint for fetching chunking comparison data"""
        storage = ChunkingComparisonStorage(app)
        comparison_data = storage.get_chunking_comparison(comparison_id)
        
        if not comparison_data:
            return jsonify({'error': 'Chunking comparison data not found or expired'}), 404
            
        return jsonify(comparison_data)
        
    @app.route('/chunking-upload')
    def chunking_upload():
        """Page for uploading a PDF to compare chunking strategies"""
        # Get the options for the chunking strategies
        chunking_strategies = ChunkingStrategy.choices()
        
        return render_template('chunking_upload.html', 
                              chunking_strategies=chunking_strategies)
    
    @app.route('/chunking-process', methods=['POST'])
    def chunking_process():
        """Process an uploaded PDF with selected chunking strategies"""
        # Check if files were uploaded
        if 'pdf_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('chunking_upload'))
            
        pdf_file = request.files['pdf_file']
        
        # If no file was selected
        if pdf_file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('chunking_upload'))
            
        # Check if the file is a PDF
        if not pdf_file.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed', 'danger')
            return redirect(url_for('chunking_upload'))
            
        # Get strategy selections
        strategy_1 = request.form.get('strategy_1')
        strategy_2 = request.form.get('strategy_2')
        
        # Check if strategies are selected
        if not strategy_1 or not strategy_2:
            flash('Please select two chunking strategies', 'danger')
            return redirect(url_for('chunking_upload'))
            
        try:
            # Initialize response data
            comparison_data = {
                'filename': secure_filename(pdf_file.filename),
                'strategy_1': strategy_1,
                'strategy_2': strategy_2,
                'strategy_name_1': ChunkingStrategy[strategy_1].display_name if strategy_1 in ChunkingStrategy.__members__ else strategy_1,
                'strategy_name_2': ChunkingStrategy[strategy_2].display_name if strategy_2 in ChunkingStrategy.__members__ else strategy_2,
                'chunks_1': [],
                'chunks_2': [],
                'stats_1': {},
                'stats_2': {},
                'error_1': None,
                'error_2': None
            }
            
            # Process the file in memory
            pdf_content = pdf_file.read()
            
            # Create BytesIO object for the PDF content
            pdf_io = io.BytesIO(pdf_content)
            
            # Extract text from PDF in memory
            pdf_text = extract_text_from_pdf_memory(pdf_io)
            
            # Count total tokens in the document
            text_token_count = count_tokens(pdf_text)
            
            # Add document stats to comparison data
            comparison_data['text_length'] = len(pdf_text)
            comparison_data['text_token_count'] = text_token_count
            
            # Process with strategy 1
            try:
                # Get parameters for strategy 1
                strategy_1_params = {}
                for key, value in request.form.items():
                    if key.startswith(f'params_1_'):
                        param_name = key.replace('params_1_', '')
                        # Convert numeric values
                        if value.isdigit():
                            value = int(value)
                        strategy_1_params[param_name] = value
                
                # Get the chunker function
                chunker_func_1 = get_chunker_function(strategy_1)
                
                # Create the parameter model
                param_model_cls = ChunkingStrategy[strategy_1].param_model
                
                # Clean up parameter names - some UI fields might not match the model exactly
                if strategy_1 == 'SEMANTIC_CHUNKING_LLAMAINDEX' and 'chunk_size' in strategy_1_params:
                    # Handle case where old parameter name was used
                    strategy_1_params['max_chunk_size'] = strategy_1_params.pop('chunk_size')
                
                params_1 = param_model_cls(**strategy_1_params)
                
                # Apply the chunking strategy
                chunks_1 = chunker_func_1(pdf_text, params_1)
                
                # Convert Chunk objects to dictionaries
                chunks_1_dicts = [
                    {
                        'chunk_text': chunk.chunk_text,
                        'metadata': chunk.metadata,
                        'char_count': chunk.char_count,
                        'token_count': chunk.token_count,
                        'chunk_id': chunk.chunk_id
                    }
                    for chunk in chunks_1
                ]
                
                # Calculate statistics
                stats_1 = calculate_chunk_statistics(chunks_1)
                
                # Store results
                comparison_data['chunks_1'] = chunks_1_dicts
                comparison_data['stats_1'] = stats_1
                
            except Exception as e:
                logging.error(f"Error processing with strategy 1: {e}")
                comparison_data['error_1'] = str(e)
                comparison_data['chunks_1'] = []
                comparison_data['stats_1'] = {
                    'total_chunks': 0,
                    'avg_chunk_length_chars': 0,
                    'avg_chunk_length_tokens': 0,
                    'min_chunk_length_tokens': 0,
                    'max_chunk_length_tokens': 0,
                    'total_chars': 0,
                    'total_tokens': 0
                }
            
            # Process with strategy 2
            try:
                # Get parameters for strategy 2
                strategy_2_params = {}
                for key, value in request.form.items():
                    if key.startswith(f'params_2_'):
                        param_name = key.replace('params_2_', '')
                        # Convert numeric values
                        if value.isdigit():
                            value = int(value)
                        strategy_2_params[param_name] = value
                
                # Get the chunker function
                chunker_func_2 = get_chunker_function(strategy_2)
                
                # Create the parameter model
                param_model_cls = ChunkingStrategy[strategy_2].param_model
                
                # Clean up parameter names - some UI fields might not match the model exactly
                if strategy_2 == 'SEMANTIC_CHUNKING_LLAMAINDEX' and 'chunk_size' in strategy_2_params:
                    # Handle case where old parameter name was used
                    strategy_2_params['max_chunk_size'] = strategy_2_params.pop('chunk_size')
                
                params_2 = param_model_cls(**strategy_2_params)
                
                # Apply the chunking strategy
                chunks_2 = chunker_func_2(pdf_text, params_2)
                
                # Convert Chunk objects to dictionaries
                chunks_2_dicts = [
                    {
                        'chunk_text': chunk.chunk_text,
                        'metadata': chunk.metadata,
                        'char_count': chunk.char_count,
                        'token_count': chunk.token_count,
                        'chunk_id': chunk.chunk_id
                    }
                    for chunk in chunks_2
                ]
                
                # Calculate statistics
                stats_2 = calculate_chunk_statistics(chunks_2)
                
                # Store results
                comparison_data['chunks_2'] = chunks_2_dicts
                comparison_data['stats_2'] = stats_2
                
            except Exception as e:
                logging.error(f"Error processing with strategy 2: {e}")
                comparison_data['error_2'] = str(e)
                comparison_data['chunks_2'] = []
                comparison_data['stats_2'] = {
                    'total_chunks': 0,
                    'avg_chunk_length_chars': 0,
                    'avg_chunk_length_tokens': 0,
                    'min_chunk_length_tokens': 0,
                    'max_chunk_length_tokens': 0,
                    'total_chars': 0,
                    'total_tokens': 0
                }
            
            # Store the comparison data
            chunking_storage = ChunkingComparisonStorage(app)
            comparison_id = chunking_storage.store_chunking_comparison(comparison_data)
            
            # Redirect to the chunking view page
            return redirect(url_for('chunking_view', comparison_id=comparison_id))
            
        except Exception as e:
            logging.error(f"Error in chunking comparison: {str(e)}")
            flash(f'Error processing comparison: {str(e)}', 'danger')
            return redirect(url_for('chunking_upload'))
            
    @app.route('/chunking-view/<comparison_id>')
    def chunking_view(comparison_id):
        """Page for reviewing chunking comparison results"""
        # Get the chunking comparison data from storage
        chunking_storage = ChunkingComparisonStorage(app)
        comparison_data = chunking_storage.get_chunking_comparison(comparison_id)
        
        if not comparison_data:
            flash('Chunking comparison data not found or expired', 'danger')
            return redirect(url_for('chunking_upload'))
            
        return render_template('chunking_view.html', comparison_data=comparison_data)
        
    # Audit Search and Scraping Routes
    @app.route('/audit-search')
    def audit_search_page():
        """Main audit search interface."""
        return render_template('audit_search.html')

    @app.route('/api/classifier/status', methods=['GET'])
    def api_classifier_status():
        """Get current classifier status."""
        try:
            from scraper.classifier import MedicaidAuditClassifier
            classifier = MedicaidAuditClassifier()
            status = classifier.get_status()
            
            return jsonify({
                'success': True,
                'status': status
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/audit-search', methods=['POST'])
    def execute_audit_search():
        """Execute search and return results."""
        # Only support days_back since Google CSE doesn't support absolute date ranges
        days_back = request.json.get('days_back', 30)
        
        try:
            service = AuditSearchService()
            results = service.search_and_classify(days_back)
            
            # Count classification errors
            classification_errors = len([r for r in results if not r.get('ai_classification', {}).get('success', True)])
            
            return jsonify({
                'success': True,
                'results': results,
                'stats': {
                    'total': len(results),
                    'audits': sum(1 for r in results 
                                 if r.get('ai_classification', {}).get('is_medicaid_audit')),
                    'duplicates': sum(1 for r in results if r.get('is_duplicate')),
                    'errors': classification_errors
                }
            })
        except Exception as e:
            logging.error(f"Search execution error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/queue/add', methods=['POST'])
    def add_to_scraping_queue():
        """Add selected items to scraping queue for review."""
        items = request.json.get('items', [])
        user_overrides = request.json.get('overrides', {})
        
        try:
            service = AuditSearchService()
            added = service.add_to_queue(items, user_overrides)
            
            return jsonify({
                'success': True,
                'added': added,
                'message': f'Added {added} reports to review queue'
            })
        except Exception as e:
            logging.error(f"Queue add error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/queue-review')
    def queue_review():
        """Page for reviewing queued items before processing."""
        try:
            service = AuditSearchService()
            pending_items = service.get_pending_review_items()
            
            return render_template('queue_review.html', 
                                pending_items=pending_items,
                                total_pending=len(pending_items))
        except Exception as e:
            logging.error(f"Queue review error: {str(e)}")
            flash('Error loading review queue', 'error')
            return redirect(url_for('dashboard'))

    @app.route('/api/queue/approve', methods=['POST'])
    def approve_queue_items():
        """Approve selected items for full AI processing."""
        item_ids = request.json.get('item_ids', [])
        
        try:
            service = AuditSearchService()
            approved = service.approve_for_processing(item_ids)
            
            return jsonify({
                'success': True,
                'approved': approved,
                'message': f'Approved {approved} reports for processing'
            })
        except Exception as e:
            logging.error(f"Queue approval error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/queue/skip', methods=['POST'])
    def skip_queue_items():
        """Skip selected items (mark as skipped)."""
        item_ids = request.json.get('item_ids', [])
        
        try:
            service = AuditSearchService()
            skipped = service.skip_items(item_ids)
            
            return jsonify({
                'success': True,
                'skipped': skipped,
                'message': f'Skipped {skipped} reports'
            })
        except Exception as e:
            logging.error(f"Queue skip error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/queue/status')
    def scraping_queue_status():
        """Get current scraping queue status."""
        try:
            stats = db.session.query(
                ScrapingQueue.status,
                func.count(ScrapingQueue.id)
            ).group_by(ScrapingQueue.status).all()
            
            recent = db.session.query(ScrapingQueue).order_by(
                ScrapingQueue.created_at.desc()
            ).limit(10).all()
            
            # Convert stats to dict and ensure all expected statuses are included
            stats_dict = dict(stats)
            for status in ['pending_review', 'pending', 'downloading', 'processing', 'completed', 'failed', 'duplicate', 'skipped']:
                if status not in stats_dict:
                    stats_dict[status] = 0
            
            return jsonify({
                'stats': stats_dict,
                'recent': [item.to_dict() for item in recent]
            })
        except Exception as e:
            logging.error(f"Queue status error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/duplicates/<path:url>')
    def check_audit_duplicates(url):
        """Get duplicate information for a URL."""
        try:
            report = db.session.query(Report).filter_by(
                original_report_source_url=url
            ).first()
            
            if report:
                return jsonify({
                    'found': True,
                    'report': {
                        'id': report.id,
                        'title': report.report_title,
                        'year': report.publication_year,
                        'month': report.publication_month,
                        'hidden': report.hidden,
                        'status': 'hidden' if report.hidden else 'visible'
                    }
                })
            
            return jsonify({'found': False})
        except Exception as e:
            logging.error(f"Duplicate check error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
