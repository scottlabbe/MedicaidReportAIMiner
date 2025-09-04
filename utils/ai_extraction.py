import os
import time
import logging
import json
from datetime import datetime
import instructor
from instructor import Mode
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from utils.token_usage_logger import TokenUsageLogger

# OpenAI model options
OPENAI_MODEL_GPT41_NANO = "gpt-4.1-nano"  # Legacy model
OPENAI_MODEL_GPT5_NANO = "gpt-5-nano"  # New model
OPENAI_MODEL_DEFAULT = OPENAI_MODEL_GPT5_NANO  # Default to latest model

# Import Gemini extraction function for AI provider choice
try:
    from utils.gemini_extraction import extract_data_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ReportData(BaseModel):
    report_title: str = Field(
        ...,
        description=
        "Exact full report title in Title Case; no quotes, abbreviations, report numbers or surrounding labels."
    )
    audit_organization: str = Field(
        ...,
        description=
        "Full legal name of the auditing organization. No abbreviations or acronyms"
    )
    publication_year: int = Field(
        ..., description="The year the report was published (4-digit)")
    publication_month: int = Field(
        ..., description="The month the report was published (1-12)")
    publication_day: Optional[int] = Field(
        None,
        description="The day the report was published (1-31), if available")
    objectives: List[str] = Field(
        [],
        description=
        "List of distinct audit objective texts. Each objective should be a separate string in the list; no numbering or labels."
    )
    findings: List[str] = Field(
        [],
        description=
        "List of distinct audit finding texts. Each finding should be a separate string in the list.(no 'Finding 1:' prefixes, numbering, or headers)"
    )
    recommendations: List[str] = Field(
        [],
        description=
        "List of distinct audit recommendation texts. Each recommendation should be a separate string in the list (no numbering or headers)."
    )
    overall_conclusion: Optional[str] = Field(
        None, description="The overall conclusion of the audit report")
    llm_insight: str = Field(
        ..., description="An AI-generated summary/insight about the report")
    potential_objective_summary: Optional[str] = Field(
        None,
        description=
        "An AI-generated summary of the audit objective, scope, and overall conclusion to build on the findings of this report to other relevant Medicaid audits"
    )
    original_report_source_url: Optional[str] = Field(
        None, description="URL to the original report, if available")
    state: str = Field(
        ...,
        description=
        "The US state code related to the agency who published report (e.g., 'NY', 'CA'). Use 'US' for federal agencies and nationwide reports."
    )
    audit_scope: str = Field(
        ..., description="The scope of the audit, including the time period.")
    extracted_keywords: List[str] = Field(
        [], description="Relevant keywords extracted from the report content")


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


def extract_data_with_openai(pdf_text, api_key, model=OPENAI_MODEL_DEFAULT):
    """
    Extract structured data from PDF text using OpenAI's API with instructor.
    
    Args:
        pdf_text: Text content of the PDF
        api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-5-nano)
        
    Returns:
        tuple: (ReportData object, AIExtractionLog object)
    """
    start_time = time.time()

    try:
        # Initialize the OpenAI client with instructor for structured output
        # Use TOOLS mode for compatibility with gpt-4.1-nano model
        client = instructor.patch(OpenAI(api_key=api_key), mode=Mode.TOOLS)

        # Prepare the system prompt
        system_prompt = """
        You are an AI assistant specialized in extracting structured information from Medicaid audit reports.
        Your task is to extract specific data points from the provided report text and format them according to the specified schema.
        Focus on accuracy and be as detailed as possible. If some information is not present in the text, leave those fields empty or null.
        
        Always generate a potential_objective_summary that builds on the reportt objectives of the audit for future audits in a concise paragraph.
        """

        # Prepare the user prompt
        user_prompt = f"""
        Please extract structured data from the following Medicaid audit report text. 
        Here's the report text:
        {pdf_text[:80000]}  # Limiting to first 80k characters for token limits
        
        If the report text is cut off, please extract as much information as possible from the provided text.
        """

        # Make the API call with structured output using instructor and specified model
        # Note: GPT-5 models only support default temperature (1.0)
        api_params = {
            "model":
            model,
            "response_model":
            ReportData,
            "messages": [{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_prompt
            }]
        }

        # Only add temperature for models that support it (GPT-4.1-nano)
        if model == OPENAI_MODEL_GPT41_NANO:
            api_params["temperature"] = 0.2

        response = client.chat.completions.create(**api_params)

        # Access the actual token usage from the raw OpenAI response
        # When using instructor, the raw response is stored in _raw_response
        actual_usage = None
        if hasattr(response, '_raw_response'):
            # Instructor wraps the response, access the original
            raw_response = response._raw_response
            if hasattr(raw_response, 'usage'):
                actual_usage = raw_response.usage
                # Log actual token usage
                TokenUsageLogger.log_extraction(
                    provider="OpenAI",
                    model=model,
                    input_tokens=actual_usage.prompt_tokens,
                    output_tokens=actual_usage.completion_tokens,
                    total_tokens=actual_usage.total_tokens,
                    input_cost=0,  # Will be calculated below
                    output_cost=0,  # Will be calculated below
                    total_cost=0,  # Will be calculated below
                    processing_time_ms=0,  # Will be set below
                    status="IN_PROGRESS",
                    report_title=response.report_title
                )
        
        # Print extraction result for debugging
        print("\n=== AI EXTRACTION RESULT ===")
        print(f"Model: {model}")
        if actual_usage:
            print(f"Actual tokens - Prompt: {actual_usage.prompt_tokens}, Completion: {actual_usage.completion_tokens}, Total: {actual_usage.total_tokens}")
        print(f"Extracted title: {response.report_title}")
        print(f"State: {response.state}")
        print("=== END OF AI EXTRACTION RESULT ===\n")

        # Add minimal validation for required fields to prevent data loss
        if not response.state:
            response.state = "US"  # Default fallback for missing state
            logging.warning("State field was missing, defaulted to 'US'")

        if not response.audit_scope:
            response.audit_scope = "Audit scope not specified in document"
            logging.warning("Audit scope field was missing, used default text")

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # ms

        # Use actual token counts if available, otherwise fall back to estimates
        if actual_usage:
            input_tokens = actual_usage.prompt_tokens
            output_tokens = actual_usage.completion_tokens
            total_tokens = actual_usage.total_tokens
        else:
            # Fallback to estimates only if actual usage is not available
            logging.warning("Could not get actual token usage from OpenAI, using estimates")
            input_tokens = len(pdf_text) // 4  # rough estimate
            output_tokens = 1000  # rough estimate
            total_tokens = input_tokens + output_tokens

        # Calculate costs based on model pricing
        if model == OPENAI_MODEL_GPT5_NANO:
            # GPT-5-nano pricing: $0.05 input / $0.40 output per 1M tokens
            input_cost = input_tokens * 0.00000005
            output_cost = output_tokens * 0.0000004
        elif model == "gpt-4o-mini":
            # GPT-4o-mini pricing: $0.15 input / $0.60 output per 1M tokens
            input_cost = input_tokens * 0.00000015
            output_cost = output_tokens * 0.0000006
        else:
            # GPT-4.1-nano or other models (legacy pricing)
            input_cost = input_tokens * 0.00001
            output_cost = output_tokens * 0.00003

        total_cost = input_cost + output_cost
        
        log = AIExtractionLog(model_name=model,
                              input_tokens=input_tokens,
                              output_tokens=output_tokens,
                              total_tokens=total_tokens,
                              input_cost=input_cost,
                              output_cost=output_cost,
                              total_cost=total_cost,
                              processing_time_ms=processing_time,
                              extraction_status="SUCCESS")
        
        # Log successful extraction with full details
        TokenUsageLogger.log_extraction(
            provider="OpenAI",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            processing_time_ms=processing_time,
            status="SUCCESS",
            report_title=response.report_title
        )
        
        # Log estimation accuracy if we have actual usage
        if actual_usage:
            estimated_input = len(pdf_text) // 4
            TokenUsageLogger.log_token_estimation_accuracy(estimated_input, input_tokens, "input")

        return (response, log)

    except Exception as e:
        logging.error(f"Error extracting data with OpenAI: {e}")

        # Calculate processing time even for error case
        processing_time = int((time.time() - start_time) * 1000)  # ms

        # Create error log with actual details
        log = AIExtractionLog(model_name=model,
                              input_tokens=0,
                              output_tokens=0,
                              total_tokens=0,
                              input_cost=0.0,
                              output_cost=0.0,
                              total_cost=0.0,
                              processing_time_ms=processing_time,
                              extraction_status="FAILURE",
                              error_details=str(e))
        
        # Log the failure
        TokenUsageLogger.log_extraction(
            provider="OpenAI",
            model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            processing_time_ms=processing_time,
            status="FAILURE",
            error=str(e)
        )

        # Re-raise with more context
        raise ValueError(f"Failed to extract data with OpenAI: {e}")


def extract_data_with_ai(
        pdf_text: str,
        provider: str = "openai",
        model: str = None) -> tuple[ReportData, AIExtractionLog]:
    """
    Extract structured data using the specified AI provider and model.
    
    Args:
        pdf_text: Text content of the PDF
        provider: AI provider to use ("openai" or "gemini")
        model: Specific model to use (for OpenAI: "gpt-5-nano", "gpt-4.1-nano")
        
    Returns:
        tuple: (ReportData object, AIExtractionLog object)
    """
    if provider.lower() == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini extraction not available. Please install required dependencies."
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables")

        return extract_data_with_gemini(pdf_text, api_key)

    elif provider.lower() == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables")

        # Use specified model or default
        openai_model = model if model else OPENAI_MODEL_DEFAULT
        return extract_data_with_openai(pdf_text, api_key, openai_model)

    else:
        raise ValueError(
            f"Unknown AI provider: {provider}. Supported providers: 'openai', 'gemini'"
        )
