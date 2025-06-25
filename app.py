import streamlit as st
import pandas as pd
import json
from components.sentiments import detect_sentiment
from components.llm_response import llm_responder
from components.feedback import log_feedback
import hashlib
from components.retrieval import Retrieval


# Using streamlit to set up a UI

st.set_page_config(page_title="Gen AI Customer Support Assistant", page_icon="🤖", layout="centered")
st.title("💬 Gen AI Customer Support Assistant")
st.write("Ask a question related to your order, refunds, or exchanges. The assistant will respond in the appropriate tone based on your message.")

# Load the vector DB 
if "db" not in st.session_state:
    invoke_vector_store = Retrieval()
    st.session_state.db = invoke_vector_store.get_db()

# create a chat interface.
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input(placeholder="e.g., Why is my refund still not processed?")

if query:
    sentiment = detect_sentiment(query)
    docs = st.session_state.db.similarity_search(query, k=3)
    if "llm" not in st.session_state:
        st.session_state.llm = llm_responder()
    
    response = st.session_state.llm.get_response(query, sentiment, docs)

    st.session_state.history.append({
        "query": query,
        "response": response,
        "sentiment": sentiment,
        "docs": docs
    })

# Display chat history
for entry in st.session_state.history:
    # User message
    with st.chat_message("user"):
        st.markdown(entry['query'])

    # Assistant response
    with st.chat_message("assistant"):
        st.markdown(entry['response'])

        # Show sources if relevant (ex: refund, tracking, customer support)
        for doc in entry['docs']:
            source = doc.metadata.get("source")
            if source:
                st.markdown(f"🔗 [Source]({source})")

        # Show feedback buttons
        query_hash = hashlib.md5(entry['query'].encode()).hexdigest()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", key=f"up_{query_hash}"):
                log_feedback(entry['query'], entry['response'], entry['sentiment'], True)
        with col2:
            if st.button("👎", key=f"down_{query_hash}"):
                log_feedback(entry['query'], entry['response'], entry['sentiment'], False)

with st.sidebar:
    if st.button("🔄 Start Over"):
        st.session_state.history = []
        st.rerun()


# Add admin section to view feedback from customer and the feedback stats.

with st.sidebar:
    st.markdown("---")
    st.markdown("**Admin Panel**")

    # Creating two radio buttons for Feedback viewing and Basic analytics of the feedback. 

    admin_tab = st.radio(
        "Choose a view:",
        ["None", "📋 Feedback Viewer", "📊 Stats"],
        index=0,
        horizontal=False,
        key="admin_tab"
    )

    # creating a dataframe with the feedback data with timestamp, sentiment, feedback, customer query and agent response. 
    if admin_tab == "📋 Feedback Viewer":
        try:
            with open("data/feedback_log.json", "r") as f:
                feedback_data = json.load(f)

            if feedback_data:
                df = pd.DataFrame(feedback_data)
                df["thumbs"] = df["thumbs_up"].apply(lambda x: "👍" if x else "👎")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values(by="timestamp", ascending=False)
                df = df[["timestamp", "sentiment", "thumbs", "query", "response"]]
                df.columns = ["Timestamp", "Sentiment", "Feedback", "Customer Query", "Agent Response"]

                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No feedback yet.")
        except FileNotFoundError:
            st.warning("feedbac_log.json file not found.")

    # Now establishing some basic analytics of the feedback. This will give the analysts or scientists a good starting point of 
    # how to improve the AI assistant's responses, and which areas the company can improve its services in.
    # 
    # Stats included: Sum of positive and negative feedback
    #                 Ratio of positive counts of feedback to total counts of feedback    
    
    elif admin_tab == "📊 Stats":
        try:
            with open("data/feedback_log.json", "r") as f:
                feedback_data = json.load(f)

            if feedback_data:
                df = pd.DataFrame(feedback_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["thumbs_up"] = df["thumbs_up"].astype(bool)

                total = len(df)
                pos = df["thumbs_up"].sum()
                neg = total - pos
                ratio = round((pos / total) * 100, 2) if total > 0 else 0
                sentiment_counts = df["sentiment"].value_counts().to_dict()

                st.metric("Total Feedback", total)
                st.metric("👍 Positive", pos)
                st.metric("👎 Negative", neg)
                st.metric("👍 Ratio", f"{ratio}%")

                st.markdown("**Sentiment Breakdown**")
                for sent, count in sentiment_counts.items():
                    st.markdown(f"- {sent.title()}: {count}")
            else:
                st.info("No feedback yet.")
        except Exception as e:
            st.error(f"Error loading stats: {e}")