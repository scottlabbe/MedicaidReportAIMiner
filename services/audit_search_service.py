# services/audit_search_service.py
import threading
from datetime import datetime
from scraper.search import MedicaidAuditSearcher
from scraper.classifier import MedicaidAuditClassifier
from models import Report, ScrapingQueue, SearchHistory
from app import db


class AuditSearchService:
    def __init__(self):
        self.searcher = MedicaidAuditSearcher()
        self.classifier = MedicaidAuditClassifier()
        
    def search_and_classify(self, days_back=30):
        """Execute search with AI classification."""
        # Search
        results = self.searcher.search(days_back=days_back, max_results=50)
        
        # Classify
        classified_results = self.classifier.classify_batch(results)
        
        # Check for duplicates
        for result in classified_results:
            result['is_duplicate'] = self._check_duplicate(result['url'])
            if result['is_duplicate']:
                result['duplicate_report'] = self._get_duplicate_info(result['url'])
        
        # Save search history
        self._save_search_history(len(classified_results), days_back)
        
        return classified_results
    
    def search_and_classify_date_range(self, start_date, end_date):
        """Execute search with AI classification for specific date range."""
        # Search with custom date range
        results = self.searcher.search_date_range(start_date=start_date, end_date=end_date, max_results=50)
        
        # Classify
        classified_results = self.classifier.classify_batch(results)
        
        # Check for duplicates
        for result in classified_results:
            result['is_duplicate'] = self._check_duplicate(result['url'])
            if result['is_duplicate']:
                result['duplicate_report'] = self._get_duplicate_info(result['url'])
        
        # Save search history with date range info
        self._save_search_history_date_range(len(classified_results), start_date, end_date)
        
        return classified_results
    
    def _check_duplicate(self, url):
        """Check if URL exists in reports or queue."""
        # Check main reports table
        existing_report = db.session.query(Report).filter_by(
            original_report_source_url=url
        ).first()
        
        if existing_report:
            return True
            
        # Check queue
        in_queue = db.session.query(ScrapingQueue).filter_by(
            url=url
        ).filter(
            ScrapingQueue.status.in_(['pending_review', 'pending', 'downloading', 'processing'])
        ).first()
        
        return in_queue is not None
    
    def _get_duplicate_info(self, url):
        """Get information about existing duplicate."""
        existing_report = db.session.query(Report).filter_by(
            original_report_source_url=url
        ).first()
        
        if existing_report:
            return {
                'id': existing_report.id,
                'title': existing_report.report_title,
                'year': existing_report.publication_year,
                'month': existing_report.publication_month
            }
        return None
    
    def add_to_queue(self, items, user_overrides=None):
        """Add items to processing queue."""
        user_overrides = user_overrides or {}
        added_count = 0
        
        for item in items:
            # Skip if already in queue or processed
            if self._check_duplicate(item['url']):
                continue
                
            # Apply user override if provided
            if item['url'] in user_overrides:
                item['ai_classification']['is_medicaid_audit'] = user_overrides[item['url']]
                item['user_override'] = True
            
            queue_item = ScrapingQueue(
                url=item['url'],
                title=item['title'],
                source_domain=item['source'],
                document_metadata=item.get('metadata', {}),
                ai_classification=item.get('ai_classification', {}),
                user_override=item.get('user_override', False)
            )
            
            db.session.add(queue_item)
            added_count += 1
        
        db.session.commit()
        
        # Note: Items are now added to 'pending_review' status
        # Background processing will start only after user approval
            
        return added_count
    
    def get_pending_review_items(self):
        """Get all items pending review."""
        return db.session.query(ScrapingQueue).filter_by(
            status='pending_review'
        ).order_by(ScrapingQueue.created_at.desc()).all()
    
    def approve_for_processing(self, item_ids):
        """Approve selected items for full AI processing."""
        approved_count = 0
        
        for item_id in item_ids:
            item = db.session.query(ScrapingQueue).filter_by(
                id=item_id,
                status='pending_review'
            ).first()
            
            if item:
                item.status = 'pending'
                approved_count += 1
        
        db.session.commit()
        
        # Start background processing for approved items
        if approved_count > 0:
            self._start_background_processing()
            
        return approved_count
    
    def skip_items(self, item_ids):
        """Skip selected items (mark as skipped)."""
        skipped_count = 0
        
        for item_id in item_ids:
            item = db.session.query(ScrapingQueue).filter_by(
                id=item_id,
                status='pending_review'
            ).first()
            
            if item:
                item.status = 'skipped'
                item.completed_at = datetime.utcnow()
                skipped_count += 1
        
        db.session.commit()
        return skipped_count

    def _start_background_processing(self):
        """Start processing queue in background thread."""
        from services.queue_processor import QueueProcessor
        processor = QueueProcessor()
        thread = threading.Thread(target=processor.process_queue)
        thread.daemon = True
        thread.start()
    
    def _save_search_history(self, results_count, days_back):
        """Save search to history."""
        history = SearchHistory(
            search_params={'days_back': days_back},
            results_count=results_count
        )
        db.session.add(history)
        db.session.commit()
    
    def _save_search_history_date_range(self, results_count, start_date, end_date):
        """Save search to history with date range."""
        history = SearchHistory(
            search_params={'start_date': start_date, 'end_date': end_date},
            results_count=results_count
        )
        db.session.add(history)
        db.session.commit()