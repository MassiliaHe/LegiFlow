from typing import List

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pylegifrance import LegifranceClient
from pylegifrance.fonds.code import Code

from pylegifrance.fonds.juri import JuriAPI

from dotenv import load_dotenv



load_dotenv()

client = LegifranceClient()
code = Code(client)
juri_api = JuriAPI(client)




class Extraction(BaseModel):
    mots_cles: List[str] = Field(..., description="2 à 5 mots")
    codes_probables: str = Field(..., description="Noms exact de codes français (ex: 'Code du travail')")


codes = ["Code de l'action sociale et des familles", "Code de l'artisanat", "Code des assurances", "Code de l'aviation civile", "Code du cinéma et de l'image animée", "Code civil", "Code de la commande publique", "Code de commerce", "Code des communes", "Code des communes de la Nouvelle-Calédonie", "Code de la consommation", "Code de la construction et de l'habitation", "Code de la défense", "Code de déontologie des architectes", "Code disciplinaire et pénal de la marine marchande", "Code du domaine de l'État", "Code du domaine de l'État et des collectivités publiques applicable à la collectivité territoriale de Mayotte", "Code du domaine public fluvial et de la navigation intérieure", "Code des douanes", "Code des douanes de Mayotte", "Code de l'éducation", "Code électoral", "Code de l'énergie", "Code de l'entrée et du séjour des étrangers et du droit d'asile", "Code de l'environnement", "Code de l'expropriation pour cause d'utilité publique", "Code de la famille et de l'aide sociale", "Code forestier (nouveau)", "Code général de la fonction publique", "Code général de la propriété des personnes publiques", "Code général des collectivités territoriales", "Code général des impôts", "Code général des impôts, annexe I", "Code général des impôts, annexe II", "Code général des impôts, annexe III", "Code général des impôts, annexe IV", "Code des impositions sur les biens et services", "Code des instruments monétaires et des médailles", "Code des juridictions financières", "Code de justice administrative", "Code de justice militaire (nouveau)", "Code de la justice pénale des mineurs", "Code de la Légion d'honneur, de la Médaille militaire et de l'ordre national du Mérite", "Livre des procédures fiscales", "Code minier", "Code minier (nouveau)", "Code monétaire et financier", "Code de la mutualité", "Code de l'organisation judiciaire", "Code du patrimoine", "Code pénal", "Code pénitentiaire", "Code des pensions civiles et militaires de retraite", "Code des pensions de retraite des marins français du commerce, de pêche ou de plaisance", "Code des pensions militaires d'invalidité et des victimes de guerre", "Code des ports maritimes", "Code des postes et des communications électroniques", "Code de procédure civile", "Code de procédure pénale", "Code des procédures civiles d'exécution", "Code de la propriété intellectuelle", "Code de la recherche", "Code des relations entre le public et l'administration", "Code de la route", "Code rural (ancien)", "Code rural et de la pêche maritime", "Code de la santé publique", "Code de la sécurité intérieure", "Code de la sécurité sociale", "Code du service national", "Code du sport", "Code du tourisme", "Code des transports", "Code du travail", "Code du travail maritime", "Code de l'urbanisme", "Code de la voirie routière"]



def build_extractor_system_prompt(codes: List[str]) -> str:
    """Return the system prompt for the extraction step (unchanged content)."""
    return f"""
    Tu es un extracteur d’informations juridiques spécialisé. 
    Ta mission est de :
    1. Identifier et extraire les mots-clés pertinents de la question de l’utilisateur.  
    2. Déterminer, parmi la liste fournie, le ou les codes juridiques qui correspondent le mieux à la question.  
    3. Préparer ces éléments (mots-clés + code identifié) pour permettre une recherche optimisée dans Légifrance.  

    Voici la liste des codes disponibles :  
    {codes}
"""


def build_summary_system_prompt(results: str) -> str:
    """Return the system prompt for the summarization step (unchanged content)."""
    return f"""
    Tu es une assistante juridique spécialisée dans la vulgarisation.  
    Ta mission est de :  
    1. Expliquer les résultats issus de la recherche sur Légifrance de manière claire et accessible, sans jargon complexe.  
    2. Traduire les termes juridiques en langage simple pour des utilisateurs non familiers avec le droit.  
    3. Fournir une réponse concise, structurée et facile à comprendre.  
    4. Résumer l’essentiel afin que l’utilisateur reparte avec une vision claire de l’information.  

    Voici les résultats de la recherche :  
    {results}
    """





def build_llm() ->ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

def format_jurisprudence_results(items) -> str:
    filtred_results = []
    for articles in items :
        article_dict = {
            "title": articles.title,
            "jurisdiction" : articles.jurisdiction,
            "text" : articles.text,
            "solution" : articles.solution
        }
        filtred_results.append(article_dict)

    filtred_results = "\n\n".join(
        [",  ".join(f"{k}: {v}" for k, v in d.items()) for d in filtred_results]
    )
    return filtred_results



def start_app():
    
    llm = build_llm()

    # Node 1 : récupérer la question de l'utilisateur 
    input_user = input('Comment puis-je vous aider :')
    print(input_user)

    # Node 2 : Récupérer les mots clés --> output (" String")

    system_prompt_extractor = build_extractor_system_prompt(codes)


    messages = [
        SystemMessage(content=system_prompt_extractor),
        HumanMessage(content=input_user)
    ]

    structured_model = llm.with_structured_output(Extraction)
    ai_msg: Extraction = structured_model.invoke(messages)
    print(ai_msg)



    # Node 3 : Faire de la recherche de légifrance avec les mots clés --> output (resultat de la recherche)
    resultats = (code.search()
                    .in_code(ai_msg.codes_probables)  # Code civil
                    .text(" ".join(ai_msg.mots_cles))
                    .execute())

    juri_results_raw = juri_api.search(" ".join(ai_msg.mots_cles))
    juri_results_strt = format_jurisprudence_results(juri_results_raw)
    

    # Node 4 : Traduire en NL les résultat de la recherche et faire  un résumé 
    system_prompt_summary = build_summary_system_prompt(juri_results_strt)
    messages_node4 = [
        SystemMessage(content=system_prompt_summary),
        HumanMessage(content=input_user)

    ]
    final_response = llm.invoke(messages_node4)
    print(final_response.content)


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

# if __name__ == "__main__":
#     start_app()
