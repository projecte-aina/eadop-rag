import logging
import os
import requests



from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAG:
    NO_ANSWER_MESSAGE: str = "Sorry, I couldn't answer your question."


    def __init__(self, hf_token, embeddings_model, model_name):


        self.model_name = model_name
        self.hf_token = hf_token
        
        # load vectore store
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': 'cpu'})
        self.vectore_store = FAISS.load_local("vectorestore", embeddings, allow_dangerous_deserialization=True)#, allow_dangerous_deserialization=True)

        logging.info("RAG loaded!")
    
    def get_context(self, instruction, number_of_contexts=1):

        context = ""


        documentos = self.vectore_store.similarity_search_with_score(instruction, k=number_of_contexts)


        for doc in documentos:

            context += doc[0].page_content

        return context
        
    def predict(self, instruction, context):

        api_key = os.getenv("HF_TOKEN")


        headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json" 
        }

        query = f"### Instruction\n{instruction}\n\n### Context\n{context}\n\n### Answer\n "


        payload = {
        "inputs": query,
        "parameters": {}
        }
        
        response = requests.post(self.model_name, headers=headers, json=payload)

        return response.json()[0]["generated_text"].split("###")[-1][8:-1]

    def get_response(self, prompt: str) -> str:
        
        context = self.get_context(prompt)

        response = self.predict(prompt, context)

        if not response:
            return self.NO_ANSWER_MESSAGE

        return response