# Aakarsh Sagar
# E-Commerce Generative AI for Customer Support

---

## This app is deployed on streamlit. Check it out here: [E-commerce Gen AI for Customer Support](https://e-commerce-gen-ai-customer-support.streamlit.app/)

---

## 1. Problem Expansion & Assumptions

### Context
Companies in e-commerce and consumer services often receive high volumes of customer service queries via chat or email. These queries span a wide range of tones: from calm inquiries to angry complaints. These queries traditionally require human agents to craft appropriately styled responses based on historical context and customer emotion.

###  Target Users
- **Internal**: Customer support automation teams  
- **External**: Customers interacting with the company via chat or helpdesk

### Assumptions
- Customers expect responses that reflect empathy and match their tone.
- An internal knowledge base of previous query-response pairs is available.
- Sentiment of customer messages can be approximated using off-the-shelf tools.
- A feedback system is useful for tuning model behavior over time.

### Measurable Outcomes
- Reduction in first-response time
- Improved Customer Satisfaction (CSAT) scores due to tone-aware responses
- Ratio of thumbs-up/thumbs-down feedback on responses
- Lower escalation rates to human agents

---

## 2. Data Requirements

### Collected Data
- Query-response pairs from prior support interactions (in JSON format)
- Metadata (e.g., source URLs or help documents)
- Real-time feedback from users via thumbs-up/down

### Additional Processed Data
- Sentiment classification (positive, neutral, upset) using `TextBlob`
- Embedding-based similarity context using `OpenAIEmbeddings`

### Privacy & Quality Handling
- No PII is stored or displayed
- All data is assumed anonymized and domain-specific
- Hash-based integrity checks ensure the knowledge base is updated only when content changes
- User feedback is logged with timestamps but without identifying info

---

## 3. Working Solution

### Repository

**GitHub Repo** [GitHub Repo - Gen AI for Customer Support](https://github.com/aakarsh-sagar/gen-ai-customer-support)

### Architecture Highlights
-  Uses **OpenAI’s GPT-4o-mini** model for generative response crafting
-  **Retrieval-Augmented Generation (RAG)** via FAISS for contextual grounding
-  **Tone-aware** responses based on sentiment analysis
-  Feedback logging with **admin dashboard** in Streamlit
-  Auto vector database update when knowledge base JSON changes
-  Modular, class-based **OOP design** in major components of the application for easy extension and testing

### Modules
- `sentiments.py`: Lightweight sentiment detection using `TextBlob`
- `vectorstore.py`: Handles loading and updating the FAISS vector DB in case of changes in the JSON dataset
- `llm_response.py`: Class wrapper around `StuffDocumentsChain` to generate tone-matched responses
- `feedback.py`: Class to store and analyze user feedback
- `retrieval.py`: Utility to decouple vectorstore loading logic
- `app.py`: Main Streamlit interface for user interaction and admin panel

---