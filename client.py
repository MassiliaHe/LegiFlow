# app.py
import os

import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage
from juri_flow import (
    Extraction,
    build_llm,
    build_extractor_system_prompt,
    build_summary_system_prompt,
    format_jurisprudence_results,
    codes,
    code,
    juri_api,
    )


st.set_page_config(page_title="Legifrance Search", page_icon="⚖️")
st.title("⚖️ Legifrance Search")


api_key = st.sidebar.text_input("GOOGLE API KEY (Gemini)", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key



def juri_chat(user_input : str) : 

    llm=build_llm()
    prompt_extractor = build_extractor_system_prompt(codes)
    extractor_messages = [SystemMessage(content=prompt_extractor), HumanMessage(content=user_input)]
    structurd_output = llm.with_structured_output(Extraction)
    ai_msg: Extraction = structurd_output.invoke(extractor_messages)

    # recherche legifrance 
    words = " ".join(ai_msg.mots_cles)
    code_search = (code.search()
                    .in_code(ai_msg.codes_probables)  # Code civil
                    .text(words)
                    .execute())

    juri_results_raw = juri_api.search(words)
    juri_result_format = format_jurisprudence_results(juri_results_raw)
    sys_prompt_sumury = build_summary_system_prompt(juri_result_format)
    summary_msgs = [SystemMessage(content=sys_prompt_sumury),HumanMessage(content=user_input) ]
    final_ansewer = llm.invoke(summary_msgs)
    
    return final_ansewer.content 



with st.form("chat_form", clear_on_submit=False):

    user_input = st.text_area(
        "Pose ta question juridique :",
        placeholder="Ex : Licenciement pour faute grave : quels délais ?",
        height=120
    )
    submitted = st.form_submit_button("Submit")

if submitted:
    if not api_key:
        st.warning("Ajoute d'abord ta GOOGLE_API_KEY dans la barre latérale.", icon="⚠️")
    elif not user_input.strip():
        st.warning("La question ne peut pas être vide.", icon="⚠️")
    else:
        with st.spinner("Analyse et recherche en cours…"):
            answer = juri_chat(user_input)
        st.markdown("### ✅ Réponse")
        st.write(answer)
