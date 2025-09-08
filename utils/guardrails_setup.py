from guardrails import Guard

def validate_output(output_text: str) -> str:
    """
    Validate and sanitize model output.
    """
    try:
        guard = Guard.from_string("""
        output:
            type: string
            constraints:
                - must not include profanity
                - must be relevant to financial fraud detection
        """)
        validated_output = guard.parse(output_text)
        return validated_output
    except Exception:
        return "⚠️ The response was filtered for safety."
