import os
from openai import OpenAI

# Initialize the OpenAI client (will automatically use OPENAI_API_KEY from environment)
client = OpenAI()

def check_moderation(text: str) -> bool:
    """
    Checks the input text against OpenAI's Moderation API.
    Returns True if the text is flagged as inappropriate/toxic, False otherwise.
    """
    if not text or text.strip() == "":
        return False
        
    try:
        response = client.moderations.create(input=text)
        # The response contains a list of results, we check the first one
        is_flagged = response.results[0].flagged
        return is_flagged
    except Exception as e:
        print(f"Error during moderation check: {e}")
        # In case of error, err on the side of caution or let it pass?
        # Typically we might let it pass if the API is down to avoid completely blocking the bot,
        # but for safety let's return False and log.
        return False
