from textblob import TextBlob

def detect_sentiment(text):
    """Function to detect the sentiment of the user query. Helps the support agent identify
    if the user is in a good mood or is upset or neutral. This will help the AI assistant to tailor the
    answers to the sentiment of the end user.

    Args:
        text (str): input string. This is a query by the user.

    Returns:
        str: based on the polarity of the query, the text model returns upset, positive, or neutral. Based on this sentiment,
        the AI assistant adjusts its response.
    """
    polarity = TextBlob(text).sentiment.polarity # type: ignore
    if polarity < -0.1:
        return "upset"
    elif polarity > 0.2:
        return "positive"
    else:
        return "neutral"