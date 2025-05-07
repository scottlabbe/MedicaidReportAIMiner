import os
import time
import logging
import json
from datetime import datetime
import instructor
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Using gpt-4.1-nano model as explicitly requested by the user
# Changed from gpt-4o as per user request
OPENAI_MODEL = "gpt-4.1-nano"

class Finding(BaseModel):
    finding_text: str = Field(..., description="The text of the finding from the audit report")
    financial_impact: Optional[float] = Field(None, description="The financial impact of this finding in dollars, if available")

class Objective(BaseModel):
    objective_text: str = Field(..., description="The text of the objective from the audit report")

class Recommendation(BaseModel):
    recommendation_text: str = Field(..., description="The text of the recommendation from the audit report")
    related_finding_index: Optional[int] = Field(None, description="The index of the related finding, if available (0-based)")

class ReportData(BaseModel):
    report_title: str = Field(..., description="The full title of the audit report, converted to standard title case (e.g., 'Annual Audit Report') even if it appears in all caps in the source.")
    audit_organization: str = Field(..., description="The organization that conducted the audit")
    publication_year: int = Field(..., description="The year the report was published (4-digit)")
    publication_month: int = Field(..., description="The month the report was published (1-12)")
    publication_day: Optional[int] = Field(None, description="The day the report was published (1-31), if available")
    objectives: List[Objective] = Field([], description="List of audit objectives")
    findings: List[Finding] = Field([], description="Comprehensive list of audit findings")
    recommendations: List[Recommendation] = Field([], description="Comprehensive list of audit recommendations")
    overall_conclusion: Optional[str] = Field(None, description="The overall conclusion of the audit report")
    llm_insight: str = Field(..., description="An AI-generated summary/insight about the report")
    potential_objective_summary: Optional[str] = Field(None, description="An AI-generated summary of the objectives")
    original_report_source_url: Optional[str] = Field(None, description="URL to the original report, if available")
    state: str = Field(..., description="The US state code related to the report (e.g., 'NY', 'CA')")
    audit_scope: str = Field(..., description="The scope of the audit, including only the time period.")
    extracted_keywords: List[str] = Field([], description="Relevant keywords extracted from the report content")

class AIExtractionLog(BaseModel):
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    processing_time_ms: int
    extraction_status: str
    error_details: Optional[str] = None

def extract_data_with_openai(pdf_text, api_key):
    """
    Extract structured data from PDF text using OpenAI's API with instructor.
    
    Args:
        pdf_text: Text content of the PDF
        api_key: OpenAI API key
        
    Returns:
        tuple: (ReportData object, AIExtractionLog object)
    """
    start_time = time.time()
    
    try:
        # Initialize the OpenAI client with instructor for structured output
        client = instructor.patch(OpenAI(api_key=api_key))
        
        # Prepare the system prompt
        system_prompt = """
        You are an AI assistant specialized in extracting structured information from Medicaid audit reports.
        Your task is to extract specific data points from the provided report text and format them according to the specified schema.
        Focus on accuracy and be as detailed as possible. If some information is not present in the text, leave those fields empty or null.
        
        Important: For audit_scope, provide a comprehensive text description of the audit's scope. This may include time periods, 
        programs examined, departments audited, or any other contextual information defining the boundaries of the audit.
        
        Always generate a potential_objective_summary that summarizes the main objectives of the audit in a concise paragraph.
        """
        
        # Prepare the user prompt
        user_prompt = f"""
        Please extract structured data from the following Medicaid audit report text. 
        Make sure to include all findings, recommendations, and objectives mentioned in the report.
        For keywords, identify 5-10 relevant terms that best represent the report content.
        Also provide an insightful summary about the report's significance and implications.
        
        For the audit_scope field, capture the full scope information including any dates, programs, 
        or organizational boundaries mentioned in the report. Format this as a comprehensive text description.
        
        Be sure to include a potential_objective_summary that concisely summarizes the audit objectives.
        
        Here's the report text:
        {pdf_text[:50000]}  # Limiting to first 50k characters for token limits
        
        If the report text is cut off, please extract as much information as possible from the provided text.
        """
        
        # Make the API call with structured output using instructor
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_model=ReportData,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2  # Low temperature for more deterministic output
        )
        
        # Print raw AI response to console
        print("\n=== RAW AI EXTRACTION RESPONSE ===")
        print(response)
        print("=== END OF RAW AI EXTRACTION RESPONSE ===\n")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        # Create AI extraction log
        # Note: In a production system, we would use the actual token counts from OpenAI's response
        # For this demonstration, we're using estimated values
        estimated_input_tokens = len(pdf_text) // 4  # rough estimate
        estimated_output_tokens = 1000  # rough estimate
        
        log = AIExtractionLog(
            model_name=OPENAI_MODEL,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            total_tokens=estimated_input_tokens + estimated_output_tokens,
            input_cost=estimated_input_tokens * 0.00001,  # rough estimate
            output_cost=estimated_output_tokens * 0.00003,  # rough estimate
            total_cost=(estimated_input_tokens * 0.00001) + (estimated_output_tokens * 0.00003),
            processing_time_ms=processing_time,
            extraction_status="SUCCESS"
        )
        
        return (response, log)
        
    except Exception as e:
        logging.error(f"Error extracting data with OpenAI: {e}")
        
        # Calculate processing time even for error case
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        # Create error log
        log = AIExtractionLog(
            model_name=OPENAI_MODEL,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0,
            output_cost=0,
            total_cost=0,
            processing_time_ms=processing_time,
            extraction_status="FAILURE",
            error_details=str(e)
        )
        
        # Re-raise with more context
        raise ValueError(f"Failed to extract data with OpenAI: {e}")
