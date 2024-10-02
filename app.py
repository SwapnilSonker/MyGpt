import streamlit as st
import os 
from langchain.prompts import PromptTemplate  
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from dotenv import load_dotenv
load_dotenv()

os.getenv("Huggingface_tokens")

st.title("First streamlit app")
display_text=st.text_input("Type a Topic/Person to get the information")

person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
match_memory=ConversationBufferMemory(input_key='person', memory_key='chat_history')

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me something about celebrity {name}"
)

llm_hgf=HuggingFaceHub(repo_id="google/flan-t5-large" ,model_kwargs={"temperature": 0, "max_length" : 64})
chain=LLMChain(
    llm=llm_hgf, prompt=first_input_prompt, verbose=True, output_key='person' , memory=person_memory)

second_prompt_template=PromptTemplate(
    input_variables=['person'],
    template="Tell me 3 major matches of this {person}"
)

chain2=LLMChain(
    llm=llm_hgf, prompt=second_prompt_template, verbose=True, output_key='matches',memory=match_memory
)

parent_chain=SequentialChain(
    chains=[chain, chain2], input_variables=['name'] , output_variables=['person', 'matches'], verbose=True
)

if display_text:
    st.write(parent_chain({
        'name':display_text
    }))
    with st.expander('Person name'):
        st.info(person_memory.buffer)
    with st.expander('Major Matches'):
        st.info(match_memory.buffer)        
