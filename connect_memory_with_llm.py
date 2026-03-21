import os

from langchain_huggingface import HuggingFaceEndpoint,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from transformers import pipeline
## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
#groq_api_key = os.getenv("GROQ_API_KEY")
# HUGGINGFACE_REPO_ID="google/flan-t5-large"
# # HF_TOKEN=os.getenv("HF_TOKEN")

# def load_llm(huggingface_repo_id):
#     llm=ChatGroq(
#                 model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
#                     temperature=0.0,
#                     groq_api_key=os.environ["GROQ_API_KEY"],
#                 )
        
  #  return llm
generator = pipeline("text-generation", model="gpt2", max_new_tokens=150)
def load_gpt2_llm():
    generator = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=150,
        pad_token_id=50256  # GPT-2 needs a pad token
    )
    return HuggingFacePipeline(pipeline=generator)
# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    print(prompt)
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(embedding_model)
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_gpt2_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
