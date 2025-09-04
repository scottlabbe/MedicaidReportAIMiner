"""
Token Usage Logger - Centralized logging for AI API token usage and costs
Provides detailed logging and monitoring capabilities for token consumption.
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TokenUsageLogger:
    """Centralized logger for tracking AI API token usage and costs"""
    
    @staticmethod
    def log_extraction(
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        input_cost: float,
        output_cost: float,
        total_cost: float,
        processing_time_ms: int,
        status: str,
        report_title: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log detailed token usage for an extraction"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "provider": provider,
            "model": model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens
            },
            "cost": {
                "input": f"${input_cost:.6f}",
                "output": f"${output_cost:.6f}",
                "total": f"${total_cost:.6f}"
            },
            "processing_time_ms": processing_time_ms,
            "status": status,
            "report_title": report_title
        }
        
        if status == "SUCCESS":
            # Log successful extraction with full details
            logger.info(
                f"AI Extraction Complete | Provider: {provider} | Model: {model} | "
                f"Tokens: {total_tokens:,} (in:{input_tokens:,}/out:{output_tokens:,}) | "
                f"Cost: ${total_cost:.6f} | Time: {processing_time_ms}ms | "
                f"Report: {report_title[:50] if report_title else 'N/A'}"
            )
            
            # Also log as structured JSON for analysis tools
            logger.debug(f"Token usage details: {json.dumps(log_entry, indent=2)}")
            
            # Log cost alert if extraction is expensive
            if total_cost > 0.01:  # Alert for extractions costing more than 1 cent
                logger.warning(
                    f"⚠️ High-cost extraction detected! Provider: {provider}, Model: {model}, "
                    f"Cost: ${total_cost:.6f}, Tokens: {total_tokens:,}"
                )
        else:
            # Log failed extraction
            log_entry["error"] = error
            logger.error(
                f"AI Extraction Failed | Provider: {provider} | Model: {model} | "
                f"Error: {error} | Time: {processing_time_ms}ms"
            )
            logger.debug(f"Failed extraction details: {json.dumps(log_entry, indent=2)}")
    
    @staticmethod
    def log_daily_summary(daily_stats: Dict[str, Any]):
        """Log daily summary of token usage and costs"""
        logger.info("=" * 60)
        logger.info("Daily Token Usage Summary")
        logger.info("=" * 60)
        
        for provider, stats in daily_stats.items():
            logger.info(
                f"{provider}: "
                f"Requests: {stats['request_count']} | "
                f"Total Tokens: {stats['total_tokens']:,} | "
                f"Total Cost: ${stats['total_cost']:.4f} | "
                f"Avg Cost/Request: ${stats['avg_cost']:.6f}"
            )
    
    @staticmethod
    def log_cost_comparison(openai_cost: float, gemini_cost: float, report_title: str):
        """Log cost comparison between providers for the same extraction"""
        
        if openai_cost > 0 and gemini_cost > 0:
            if gemini_cost < openai_cost:
                savings = ((openai_cost - gemini_cost) / openai_cost) * 100
                logger.info(
                    f"💰 Cost Comparison for '{report_title[:50]}': "
                    f"Gemini ${gemini_cost:.6f} vs OpenAI ${openai_cost:.6f} "
                    f"(Gemini {savings:.1f}% cheaper)"
                )
            else:
                premium = ((gemini_cost - openai_cost) / openai_cost) * 100
                logger.info(
                    f"💰 Cost Comparison for '{report_title[:50]}': "
                    f"OpenAI ${openai_cost:.6f} vs Gemini ${gemini_cost:.6f} "
                    f"(OpenAI {premium:.1f}% cheaper)"
                )
    
    @staticmethod
    def log_token_estimation_accuracy(estimated: int, actual: int, token_type: str = "total"):
        """Log the accuracy of token estimation vs actual usage"""
        
        if estimated > 0:
            accuracy = (abs(actual - estimated) / estimated) * 100
            if accuracy > 20:  # More than 20% off
                logger.warning(
                    f"Token estimation was {accuracy:.1f}% off for {token_type} tokens. "
                    f"Estimated: {estimated:,}, Actual: {actual:,}"
                )
            else:
                logger.debug(
                    f"Token estimation accuracy for {token_type}: {100-accuracy:.1f}% "
                    f"(Estimated: {estimated:,}, Actual: {actual:,})"
                )
    
    @staticmethod
    def format_token_report(ai_log) -> str:
        """Format a detailed token usage report for display"""
        
        report = []
        report.append("=" * 50)
        report.append("Token Usage Report")
        report.append("=" * 50)
        report.append(f"Model: {ai_log.model_name}")
        report.append(f"Status: {ai_log.extraction_status}")
        report.append("")
        report.append("Token Counts:")
        report.append(f"  Input:  {ai_log.input_tokens:>10,} tokens")
        report.append(f"  Output: {ai_log.output_tokens:>10,} tokens")
        report.append(f"  Total:  {ai_log.total_tokens:>10,} tokens")
        report.append("")
        report.append("Cost Breakdown:")
        report.append(f"  Input Cost:  ${ai_log.input_cost:>10.6f}")
        report.append(f"  Output Cost: ${ai_log.output_cost:>10.6f}")
        report.append(f"  Total Cost:  ${ai_log.total_cost:>10.6f}")
        report.append("")
        report.append(f"Processing Time: {ai_log.processing_time_ms:,}ms")
        
        if ai_log.error_details:
            report.append(f"Error: {ai_log.error_details}")
        
        report.append("=" * 50)
        
        return "\n".join(report)