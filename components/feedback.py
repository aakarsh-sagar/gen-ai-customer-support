import json
from datetime import datetime
import os

def log_feedback(query, response, sentiment, thumbs_up=True):
    feedback = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response": response,
        "sentiment": sentiment,
        "thumbs_up": thumbs_up
    }
    if not os.path.exists("feedback_log.json"):
        with open("feedback_log.json", "w") as f:
            json.dump([feedback], f, indent=2)
    else:
        with open("feedback_log.json", "r+") as f:
            data = json.load(f)
            data.append(feedback)
            f.seek(0)
            json.dump(data, f, indent=2)