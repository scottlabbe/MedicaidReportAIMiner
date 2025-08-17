#!/usr/bin/env python3
"""Import keyword mappings from CSV to database"""

import csv
import os
from app import app, db
from models import KeywordMapping

def import_keyword_mappings():
    """Import all keyword mappings from CSV file"""
    
    with app.app_context():
        # Clear existing data
        KeywordMapping.query.delete()
        db.session.commit()
        
        # Import from CSV
        imported_count = 0
        with open('keyword_mapping_clean.csv', 'r') as f:
            reader = csv.DictReader(f)
            
            batch = []
            for row in reader:
                # Handle possible BOM or encoding issues
                canonical = row.get('canonical_keyword') or row.get('\ufeffcanonical_keyword') 
                slug = row.get('slug')
                variation = row.get('variation')
                
                mapping = KeywordMapping(
                    canonical_keyword=canonical,
                    slug=slug,
                    variation=variation
                )
                batch.append(mapping)
                
                # Insert in batches of 100
                if len(batch) >= 100:
                    db.session.add_all(batch)
                    db.session.commit()
                    imported_count += len(batch)
                    print(f"Imported {imported_count} mappings...")
                    batch = []
            
            # Insert remaining batch
            if batch:
                db.session.add_all(batch)
                db.session.commit()
                imported_count += len(batch)
        
        print(f"Successfully imported {imported_count} keyword mappings!")
        return imported_count

if __name__ == '__main__':
    import_keyword_mappings()