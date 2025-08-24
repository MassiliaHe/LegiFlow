# app.py
import os

import streamlit as st

from client import juri_chat


USER = "user"
ASSISTANT = "assistant"


st.set_page_config(page_title="Legifrance Search", page_icon="⚖️")
st.title("⚖️ Legifrance Search")


api_key = st.sidebar.text_input("GOOGLE API KEY (Gemini)", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.info("Please add your GOOGLE API KEY")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": ASSISTANT, "content": "How can I help you?"}]
    

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_input = st.chat_input(placeholder="Posez vos questions sur le droit français", accept_file="multiple", file_type=["pdf", "png", "csv", "jpg", "jpeg"])

if user_input:

    input_text = user_input["text"]
    input_files = user_input["files"]


    st.session_state.messages.append({"role": USER, "content": input_text})
    st.chat_message(USER).write(input_text)

    final_answer = juri_chat(input_text) # files
    st.session_state.messages.append({"role": ASSISTANT, "content": final_answer})
    st.chat_message(ASSISTANT).write(final_answer)
