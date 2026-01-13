import streamlit as st
import time
import os

st.set_page_config(page_title="AI Chatbot")

# Check Access
if 'user' not in st.session_state:
    st.session_state.user = "Guest"


st.title(" AI Chat Assistant")
st.caption("Ask me anything about your courses!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi {st.session_state.user}! How can I help you learn today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call RAG API
        try:
            import requests
            
            # Prepare request payload
            payload = {
                "message": prompt,
                "sessionId": st.session_state.get('user', 'Guest'),
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            }
            
            with st.spinner("Thinking..."):
                api_url = f"{os.getenv('RAG_API_URL', 'http://localhost:5002')}/chat"
                response = requests.post(api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    full_response = data.get("response", "I'm sorry, I couldn't generate a response.")
                    sources = data.get("sources", [])
                    
                    # Display response
                    message_placeholder.markdown(full_response)
                    
                    # Display sources if any
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.markdown(f"- {source.get('content', '')[:150]}...")
                                
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
                    full_response = "I encountered an error while processing your request."
                    
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Could not connect to the AI Chatbot service. Is it running?")
            full_response = "Service unavailable."
        except Exception as e:
            message_placeholder.error(f"An error occurred: {str(e)}")
            full_response = "An unexpected error occurred."
            
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
