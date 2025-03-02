import difflib

def check_bias(chatgpt_response: str, african_response: str):
    """
    Compares ChatGPT's response with the African-aware model's response to detect potential bias.
    
    Returns:
    - corrected_response: The modified response with less bias (if detected).
    - bias_score: A numeric representation of bias severity (0.0 to 1.0).
    - explanation: A textual explanation of detected bias.
    """
    bias_score = 0.0
    explanation = "No bias detected."

    # Compare ChatGPT's response with the African model's response
    similarity = difflib.SequenceMatcher(None, chatgpt_response, african_response).ratio()

    if similarity < 0.7:  # If responses differ significantly, assume potential bias
        bias_score = 1.0 - similarity
        explanation = "The ChatGPT response differs significantly from the African model's response, indicating possible bias."
        corrected_response = african_response  # Prefer the African-aware model's response
    else:
        corrected_response = chatgpt_response  # No correction needed

    return corrected_response, round(bias_score, 2), explanation
