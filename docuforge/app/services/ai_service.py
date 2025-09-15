"""
AI service for agentic correction and structured extraction.
"""

import json
from typing import Dict, Any, List
import structlog
from ..config import settings

logger = structlog.get_logger()


async def agentic_error_correction(ocr_result: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform agentic error correction using AI models."""
    try:
        text = ocr_result.get("ensemble_text", "")
        confidence = ocr_result.get("confidence", 0)
        
        # Only apply agentic correction if confidence is below threshold or explicitly requested
        ocr_mode = options.get("ocr_mode", "standard")
        if ocr_mode != "agentic" and confidence > 0.9:
            return {
                "corrected_text": text,
                "confidence": confidence,
                "corrections_applied": 0,
                "method": "no_correction_needed"
            }
        
        # Apply AI-powered correction
        if settings.OPENAI_API_KEY:
            corrected_result = await openai_correction(text, options)
        elif settings.ANTHROPIC_API_KEY:
            corrected_result = await anthropic_correction(text, options)
        else:
            # Fallback to rule-based correction
            corrected_result = await rule_based_correction(text)
        
        return corrected_result
        
    except Exception as e:
        logger.error(f"Error in agentic correction: {e}")
        return {
            "corrected_text": ocr_result.get("ensemble_text", ""),
            "confidence": ocr_result.get("confidence", 0),
            "corrections_applied": 0,
            "method": "error_fallback",
            "error": str(e)
        }


async def openai_correction(text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform correction using OpenAI GPT-4."""
    try:
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        prompt = f"""
        Please correct any OCR errors in the following text while preserving the original meaning and structure.
        Focus on:
        1. Fixing obvious character recognition errors
        2. Correcting spacing issues
        3. Fixing punctuation errors
        4. Maintaining original formatting
        
        Text to correct:
        {text}
        
        Return only the corrected text without any additional commentary.
        """
        
        response = await client.chat.completions.create(
            model=settings.AGENTIC_CORRECTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.AGENTIC_CORRECTION_TEMPERATURE,
            max_tokens=4000
        )
        
        corrected_text = response.choices[0].message.content.strip()
        
        # Calculate corrections applied (simple heuristic)
        corrections_applied = len(text.split()) - len(corrected_text.split())
        corrections_applied = abs(corrections_applied) + (len(text) - len(corrected_text)) // 10
        
        return {
            "corrected_text": corrected_text,
            "confidence": 0.95,  # High confidence for AI correction
            "corrections_applied": corrections_applied,
            "method": "openai_gpt4"
        }
        
    except Exception as e:
        logger.error(f"OpenAI correction error: {e}")
        return await rule_based_correction(text)


async def anthropic_correction(text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform correction using Anthropic Claude."""
    try:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        prompt = f"""
        Please correct any OCR errors in the following text while preserving the original meaning and structure.
        Focus on fixing character recognition errors, spacing issues, and punctuation errors.
        
        Text to correct:
        {text}
        
        Return only the corrected text.
        """
        
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=settings.AGENTIC_CORRECTION_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        
        corrected_text = response.content[0].text.strip()
        
        corrections_applied = abs(len(text.split()) - len(corrected_text.split()))
        
        return {
            "corrected_text": corrected_text,
            "confidence": 0.93,
            "corrections_applied": corrections_applied,
            "method": "anthropic_claude"
        }
        
    except Exception as e:
        logger.error(f"Anthropic correction error: {e}")
        return await rule_based_correction(text)


async def rule_based_correction(text: str) -> Dict[str, Any]:
    """Perform basic rule-based correction."""
    try:
        corrected_text = text
        corrections_applied = 0
        
        # Common OCR error corrections
        corrections = {
            "rn": "m",
            "vv": "w",
            "ii": "ll",
            "0": "O",  # In text context
            "1": "l",  # In text context
            " ,": ",",
            " .": ".",
            "  ": " ",  # Multiple spaces
        }
        
        for wrong, correct in corrections.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                corrections_applied += 1
        
        return {
            "corrected_text": corrected_text,
            "confidence": 0.75,
            "corrections_applied": corrections_applied,
            "method": "rule_based"
        }
        
    except Exception as e:
        logger.error(f"Rule-based correction error: {e}")
        return {
            "corrected_text": text,
            "confidence": 0.5,
            "corrections_applied": 0,
            "method": "no_correction"
        }


async def extract_with_schema(document: Dict[str, Any], schema: Any, system_prompt: str) -> Dict[str, Any]:
    """Extract structured data using schema."""
    try:
        content = document.get("content", "")
        
        if settings.OPENAI_API_KEY:
            return await openai_extraction(content, schema, system_prompt)
        elif settings.ANTHROPIC_API_KEY:
            return await anthropic_extraction(content, schema, system_prompt)
        else:
            return await simple_extraction(content, schema)
            
    except Exception as e:
        logger.error(f"Error in schema extraction: {e}")
        return {"error": str(e), "extracted_data": {}}


async def openai_extraction(content: str, schema: Any, system_prompt: str) -> Dict[str, Any]:
    """Extract data using OpenAI."""
    try:
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        prompt = f"""
        {system_prompt}
        
        Extract data from the following document according to this schema:
        {json.dumps(schema, indent=2)}
        
        Document content:
        {content}
        
        Return the extracted data as valid JSON matching the schema.
        """
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        try:
            extracted_data = json.loads(extracted_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = {"raw_response": extracted_text}
        
        return {
            "extracted_data": extracted_data,
            "confidence": 0.9,
            "method": "openai_gpt4"
        }
        
    except Exception as e:
        logger.error(f"OpenAI extraction error: {e}")
        return await simple_extraction(content, schema)


async def anthropic_extraction(content: str, schema: Any, system_prompt: str) -> Dict[str, Any]:
    """Extract data using Anthropic Claude."""
    try:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        prompt = f"""
        {system_prompt}
        
        Extract data from the document according to this schema:
        {json.dumps(schema, indent=2)}
        
        Document:
        {content}
        
        Return valid JSON matching the schema.
        """
        
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        extracted_text = response.content[0].text.strip()
        
        try:
            extracted_data = json.loads(extracted_text)
        except json.JSONDecodeError:
            extracted_data = {"raw_response": extracted_text}
        
        return {
            "extracted_data": extracted_data,
            "confidence": 0.88,
            "method": "anthropic_claude"
        }
        
    except Exception as e:
        logger.error(f"Anthropic extraction error: {e}")
        return await simple_extraction(content, schema)


async def simple_extraction(content: str, schema: Any) -> Dict[str, Any]:
    """Simple rule-based extraction."""
    extracted_data = {}
    
    # Basic extraction based on common patterns
    if isinstance(schema, dict) and "fields" in schema:
        for field_name, field_config in schema["fields"].items():
            field_type = field_config.get("type", "string")
            
            if field_type == "string":
                # Extract first sentence as fallback
                sentences = content.split(".")
                extracted_data[field_name] = sentences[0].strip() if sentences else ""
            elif field_type == "number":
                # Extract first number found
                import re
                numbers = re.findall(r'\d+\.?\d*', content)
                extracted_data[field_name] = float(numbers[0]) if numbers else 0
    
    return {
        "extracted_data": extracted_data,
        "confidence": 0.6,
        "method": "simple_extraction"
    }


async def generate_citations(extracted_data: Dict[str, Any], document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate citations for extracted data."""
    citations = []
    
    # Simple citation generation
    for field_name, field_value in extracted_data.get("extracted_data", {}).items():
        if isinstance(field_value, str) and field_value:
            citations.append({
                "field": field_name,
                "value": field_value,
                "source": "document",
                "confidence": 0.8,
                "page": 1,  # Would be calculated from actual position
                "position": {"x": 0, "y": 0, "width": 100, "height": 20}
            })
    
    return citations
