# app.py
import os
import streamlit as st

from legiflow.backend.client import juri_chat
from legiflow.backend.juri_flow import build_llm
from legiflow.backend.process_files import process_user_file

USER = "user"
ASSISTANT = "assistant"


st.set_page_config(page_title="Legifrance Search", page_icon="⚖️")
st.title("⚖️ Legifrance Search")


api_key = st.sidebar.text_input("GOOGLE API KEY (Gemini)", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = build_llm()
else:
    st.info("Please add your GOOGLE API KEY")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": ASSISTANT, "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if user_input := st.chat_input(placeholder="Posez vos questions sur le droit français", accept_file="multiple"):

    input_text = (user_input.text or "").strip() 
    st.session_state.messages.append({"role": USER, "content": input_text})
    st.chat_message(USER).write(input_text)


    files_in_chat = process_user_file(user_input)   
    final_answer = juri_chat(input_text + "\n" + "\n".join(files_in_chat))
    st.session_state.messages.append({"role": ASSISTANT, "content": final_answer})
    st.chat_message(ASSISTANT).write(final_answer)

