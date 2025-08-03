# scraper/search.py
"""
Google Custom Search integration for finding Medicaid audit PDFs.
"""
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

from scraper.classifier import MedicaidAuditClassifier
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import yaml

# Load environment variables
load_dotenv()

console = Console()


class MedicaidAuditSearcher:
   """Handles searching for Medicaid audit PDFs using Google Custom Search API."""
   
   def __init__(self):
       """Initialize the searcher with API credentials and config."""
       self.api_key = os.getenv("GOOGLE_API_KEY")
       self.cse_id = os.getenv("GOOGLE_CSE_ID")
       
       if not self.api_key or not self.cse_id:
           raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in .env file")
       
       # Load config
       with open("config.yaml", "r") as f:
           self.config = yaml.safe_load(f)
       
       # Build the service
       self.service = build("customsearch", "v1", developerKey=self.api_key)
   
   def build_query(self) -> str:
    """Build query using sites from config file."""
    # Get all audit sites from config
    federal_sites = self.config.get('search', {}).get('audit_sites', {}).get('federal', [])
    state_sites = self.config.get('search', {}).get('audit_sites', {}).get('state', [])
    
    # Combine all sites
    all_sites = federal_sites + state_sites
    
    # Build site: operators
    site_operators = [f"site:{site}" for site in all_sites]
    sites_query = " OR ".join(site_operators)
    
    # Build final query
    query = f'filetype:pdf ({sites_query}) Medicaid audit'
    
    console.print(f"[dim]Searching {len(all_sites)} audit sites[/dim]")
    
    return query
    
   def is_likely_audit(self, result: Dict[str, Any]) -> bool:
       """Quick filter to identify likely audit documents."""
       title_lower = result['title'].lower()
       url_lower = result['url'].lower()
       snippet_lower = result.get('snippet', '').lower()
       
       # Must mention medicaid somewhere
       if not any('medicaid' in text for text in [title_lower, url_lower, snippet_lower]):
           return False
       
       # Exclude obvious non-audits
       exclude_terms = ['manual', 'guide', 'form', 'application', 'faq', 
                       'enrollment', 'provider directory', 'bulletin', 'newsletter']
       if any(term in title_lower for term in exclude_terms):
           return False
       
       # Bonus points for audit-related terms (but not required)
       audit_terms = ['audit', 'review', 'investigation', 'findings', 'report']
       has_audit_term = any(term in title_lower for term in audit_terms)
       
       # Accept if from .gov and mentions Medicaid (even without explicit audit terms)
       return True
   
   def search(self, days_back: int = 30, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Search for Medicaid audit PDFs.
    
    Args:
        days_back: Number of days to search back from today
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with PDF information
    """
    query = self.build_query()
    console.print(f"[bold blue]Search Query:[/bold blue] {query}")
    
    # Calculate date restriction
    date_restrict = f"d{days_back}" if days_back else None
    
    results = []
    pdf_results = []
    
    try:
        # Google CSE returns max 10 results per request
        # So we need to paginate for more results
        for start_index in range(1, min(max_results + 1, 101), 10):
            request = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                dateRestrict=date_restrict,
                start=start_index,
                num=min(10, max_results - len(results))
            )
            
            response = request.execute()
            
            if "items" in response:
                results.extend(response["items"])
                
                # Filter for actual PDFs and likely audits
                for item in response["items"]:
                    if item.get("link", "").lower().endswith(".pdf"):
                        # Enhanced metadata capture
                        pdf_result = {
                            # Basic fields (existing)
                            "title": item.get("title", "Unknown"),
                            "url": item["link"],
                            "snippet": item.get("snippet", ""),
                            "source": item.get("displayLink", ""),
                            
                            # New metadata fields
                            "mime_type": item.get("mime", ""),
                            "file_format": item.get("fileFormat", ""),
                            "formatted_url": item.get("formattedUrl", ""),
                            
                            # Extract metatags if available
                            "metadata": {
                                "author": None,
                                "creation_date": None,
                                "subject": None,
                                "creator": None
                            },
                            
                            # Thumbnail if available
                            "thumbnail_url": None
                        }
                        
                        # Safely extract pagemap data
                        if "pagemap" in item:
                            # Get metatags
                            metatags = item["pagemap"].get("metatags", [{}])[0]
                            pdf_result["metadata"]["author"] = metatags.get("author", "")
                            pdf_result["metadata"]["creation_date"] = metatags.get("creationdate", "")
                            pdf_result["metadata"]["subject"] = metatags.get("subject", "")
                            pdf_result["metadata"]["creator"] = metatags.get("creator", "")
                            
                            # Get thumbnail
                            thumbnails = item["pagemap"].get("cse_thumbnail", [])
                            if thumbnails:
                                pdf_result["thumbnail_url"] = thumbnails[0].get("src", "")
                        
                        # Apply our pragmatic filter
                        if self.is_likely_audit(pdf_result):
                            pdf_results.append(pdf_result)
            
            # Stop if we have enough results
            if len(pdf_results) >= max_results:
                break
                
            # Stop if no more results
            if "items" not in response or len(response["items"]) < 10:
                break
                
    except HttpError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        # Don't expose API keys in error messages
        error_msg = str(e).replace(self.api_key, "[API_KEY_HIDDEN]")
        console.print(f"[yellow]Error details:[/yellow] {error_msg}")
        raise
    
    console.print(f"[green]Found {len(pdf_results)} likely Medicaid audit PDFs[/green]")
    
    # Debug: Show what metadata we captured
    for idx, result in enumerate(pdf_results[:3]):  # Show first 3
        if result["metadata"]["author"] or result["metadata"]["creation_date"]:
            console.print(f"[dim]Document {idx+1} metadata:[/dim]")
            if result["metadata"]["author"]:
                console.print(f"  Author: {result['metadata']['author']}")
            if result["metadata"]["creation_date"]:
                console.print(f"  Created: {result['metadata']['creation_date']}")
    
    return pdf_results
   
   def display_results(self, results: List[Dict[str, Any]]) -> None:
    """Display search results in a nice table."""
    if not results:
        console.print("[yellow]No PDF results found.[/yellow]")
        return
    
    table = Table(title=f"Found {len(results)} Likely Medicaid Audit PDFs")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Source", style="green")
    table.add_column("Author", style="yellow")  # New column
    table.add_column("URL", style="blue", no_wrap=False)
    
    for idx, result in enumerate(results, 1):
        # Mark if title contains "audit"
        title = result["title"]
        if "audit" in title.lower():
            title = f"[bold]{title}[/bold]"
        
        # Get author from metadata
        author = result.get("metadata", {}).get("author", "")
        if not author:
            author = "[dim]Unknown[/dim]"
        else:
            author = author[:30] + "..." if len(author) > 30 else author
        
        table.add_row(
            str(idx),
            title[:60] + "..." if len(result["title"]) > 60 else title,
            result["source"],
            author,
            result["url"][:40] + "..." if len(result["url"]) > 40 else result["url"]
        )
    
    console.print(table)


def main():
    """Run a test search with AI classification."""
    try:
        searcher = MedicaidAuditSearcher()
        
        console.print("[bold]🔍 Medicaid Audit PDF Search - Government Auditor Sites[/bold]\n")
        
        # Search
        results = searcher.search(days_back=30, max_results=20)
        
        # Classify with AI
        try:
            classifier = MedicaidAuditClassifier()
            classified_results = classifier.classify_batch(results)
            results = classified_results  # Update results with classification
        except Exception as e:
            console.print(f"[yellow]Warning: AI classification failed: {e}[/yellow]")
            console.print("[yellow]Continuing with unclassified results...[/yellow]")
            # Add empty classification to maintain consistency
            for result in results:
                result['ai_summary_is_medicaid'] = None
        
        # Display results
        searcher.display_results(results)
        
        # Show classification summary if available
        if results and 'ai_summary_is_medicaid' in results[0]:
            display_classification_summary(results)
        
        # Show metrics
        console.print(f"\n[bold]Metrics:[/bold]")
        console.print(f"• Total PDFs found: {len(results)}")
        console.print(f"• Unique sources: {len(set(r['source'] for r in results))}")
        console.print(f"• Search timeframe: Last 30 days")
        
        # Count likely audits
        if results and 'ai_summary_is_medicaid' in results[0] and results[0]['ai_summary_is_medicaid']:
            likely_audits = [r for r in results 
                           if r.get('ai_summary_is_medicaid', {}).get('is_medicaid_audit', False)]
            console.print(f"• [bold green]Likely Medicaid audits (AI): {len(likely_audits)}[/bold green]")
        
        # Save results
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        
        # Save all results
        with open(f"reports/search_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save filtered audits if AI classification worked
        if results and 'ai_summary_is_medicaid' in results[0] and results[0]['ai_summary_is_medicaid']:
            likely_audits = [r for r in results 
                           if r.get('ai_summary_is_medicaid', {}).get('is_medicaid_audit', False)]
            with open(f"reports/likely_audits_{timestamp}.json", "w") as f:
                json.dump(likely_audits, f, indent=2)
            console.print(f"\n[green]✓ Results saved to reports/search_results_{timestamp}.json[/green]")
            console.print(f"[green]✓ Likely audits saved to reports/likely_audits_{timestamp}.json[/green]")
        else:
            console.print(f"\n[green]✓ Results saved to reports/search_results_{timestamp}.json[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


def display_classification_summary(results: List[Dict[str, Any]]) -> None:
    """Display AI classification summary."""
    console.print("\n[bold]AI Classification Summary:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Title", no_wrap=False)
    table.add_column("Is Audit?", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Type")
    
    for idx, result in enumerate(results, 1):
        ai_class = result.get('ai_summary_is_medicaid', {})
        if ai_class:
            is_audit = "✓" if ai_class.get('is_medicaid_audit') else "✗"
            confidence = f"{ai_class.get('confidence', 0):.0%}"
            doc_type = ai_class.get('document_type', 'unknown')
            
            # Color coding
            if ai_class.get('is_medicaid_audit'):
                is_audit = f"[green]{is_audit}[/green]"
                row_style = "green"
            else:
                is_audit = f"[red]{is_audit}[/red]"
                row_style = None
                
            table.add_row(
                str(idx),
                result['title'][:60] + "...",
                is_audit,
                confidence,
                doc_type,
                style=row_style
            )
    
    console.print(table)


if __name__ == "__main__":
    main()