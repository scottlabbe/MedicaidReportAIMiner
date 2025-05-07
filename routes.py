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
from utils.chunking_strategies import ChunkingStrategy, get_chunker_function, calculate_chunk_statistics
from utils.chunking_storage import ChunkingComparisonStorage

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
        
        reports = Report.query.order_by(Report.created_at.desc()).paginate(page=page, per_page=per_page)
        
        return render_template('reports.html', reports=reports)
        
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
    
    # PDF serving endpoint removed as we no longer store PDFs on disk
