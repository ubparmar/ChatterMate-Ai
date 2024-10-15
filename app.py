# app.py

import streamlit as st
from chatbot.model import get_response

st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")
st.title("🤖 AI Chatbot using Hugging Face Inference API")

# Initialize conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display existing messages
for message in st.session_state['messages']:
    with st.chat_message(message['role'].lower()):
        st.markdown(message['content'])

# Get user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state['messages'].append({"role": "User", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            assistant_response = get_response(
                prompt,
                st.session_state['messages'],
                model_name="facebook/blenderbot-400M-distill"  # Change model as desired
            )
        except Exception as e:
            assistant_response = "I'm sorry, but I'm unable to process your request at the moment."
            print(f"Error: {e}")

        # Display assistant response
        message_placeholder.markdown(assistant_response)

    # Add assistant response to history
    st.session_state['messages'].append({"role": "Assistant", "content": assistant_response})
