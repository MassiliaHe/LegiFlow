# app.py
import os
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from client import juri_chat
from juri_flow import build_llm

USER = "user"
ASSISTANT = "assistant"

class Metadata(BaseModel):
    first_name: str
    last_name: str
    document_type: str
    ocr_data: str

    def to_path(self):
        return f"{self.document_type}/{self.first_name}_{self.last_name}"


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

    metadata_prompt = """
    T'es un expert en extraction d'informations. T'as aussi la fonctionnalité OCR en toi. 
    Tu va recevoir un document, et tu dois le lire attentivement. 

    Tu dois extraire : 
        - Le nom et le prénom de la personne à qui appartient le document. 
        - Le type du document. Tu choisi impérativement l'un des types au dessous. 
        - Tu revois le resultat de la foncationalité OCR.
    
    Types de document (choisir un seul):
    - Procédure
    - État civil
    - Témoignages
    - Impôts
    - Intégration professionnelle
    - Domicile

    """

    files_in_chat = []
    for f in user_input.files:
        if bytes_data := f.read():
            human_content = [
                {"type": "text", "text": metadata_prompt}, 
                {"type": "media", "data": bytes_data, "mime_type": f.type}
            ]
            structured_output = llm.with_structured_output(Metadata)
            answer: Metadata = structured_output.invoke([HumanMessage(content=human_content)])
            relative_path = answer.to_path()
            print("final_answer : ", answer)
            metadata_json = answer.model_dump_json()
            files_in_chat.append(metadata_json) 
            BASE_DIR = Path("/home/massilia/Documents")
            # get path
            final_dir = BASE_DIR / relative_path
            final_dir.mkdir(parents=True, exist_ok=True) 
            # use the bytes, file name and file type to save the file in the predicted path.
            file_name = f.name
            save_path = final_dir / file_name
            with open(save_path, "wb") as out:
                out.write(bytes_data)

            print("Saved to:", save_path)


       
    final_answer = juri_chat(input_text + "\n" + "\n".join(files_in_chat))
    st.session_state.messages.append({"role": ASSISTANT, "content": final_answer})
    st.chat_message(ASSISTANT).write(final_answer)

