from langchain_core.messages import SystemMessage, HumanMessage
from backend.juri_flow import (
    Extraction,
    build_llm,
    build_extractor_system_prompt,
    build_summary_system_prompt,
    format_jurisprudence_results,
    codes,
    code,
    juri_api,
    )



def juri_chat(user_input : str) : 

    llm=build_llm()
    prompt_extractor = build_extractor_system_prompt(codes)
    extractor_messages = [SystemMessage(content=prompt_extractor), HumanMessage(content=user_input)]
    structurd_output = llm.with_structured_output(Extraction)
    ai_msg: Extraction = structurd_output.invoke(extractor_messages)

    # recherche legifrance 
    words = " ".join(ai_msg.mots_cles)
    # code_search = (code.search()
    #                 .in_code(ai_msg.codes_probables)  # Code civil
    #                 .text(words)
    #                 .execute())

    juri_results_raw = juri_api.search(words)
    juri_result_format = format_jurisprudence_results(juri_results_raw)
    sys_prompt_sumury = build_summary_system_prompt(juri_result_format)
    summary_msgs = [SystemMessage(content=sys_prompt_sumury),HumanMessage(content=user_input) ]
    final_ansewer = llm.invoke(summary_msgs)
    
    return final_ansewer.content

