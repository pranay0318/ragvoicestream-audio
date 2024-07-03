import os
import requests
import pyaudio
from openai import OpenAI
import base64
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key)

llm = ChatOpenAI(model="gpt-4o")
loader = CSVLoader(file_path="./laptops.csv")
data = loader.load()
vector = FAISS.from_documents(data, OpenAIEmbeddings())

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert sales assistant at a laptop store. Your job is to recommend a few laptops to the 
    customer based on their requirements.
    If the user is asking something other than buying laptops, let them know that this is a laptop store and purchasing 
    laptops is the only thing that can be done here. 
    Try to fit in the customer to a user persona like student, teacher, gamer, 
    business professional, researcher, teacher, casual everyday user, content creator etc. to recommend laptops. You 
    don't have to stick to the above personas strictly but this is a good guide. It is very important that you do not 
    mention to the user that you're trying to fit them into these personas. It is extremely important to ask user 
    questions to get more information about user's requirements and work pattern if you do not have enough 
    information to make an informed decision.It is absolutely mandatory to use only the below laptop products with 
    their detailed specification as mentioned in the inventory. Context:

<context>
{context}
</context>
         """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retriever.search_kwargs = {'k': 10}

history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the "
             "conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'onyx',
        response_format = 'pcm',
        input=text,
    ) as response:
        silence_threshold=0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def llm_w_stream(message, chat_history):
    text = retrieval_chain.invoke({"input": f"{message}", "chat_history": chat_history})["answer"]
    return text
    # sentence = ''
    # sentences = []
    # sentence_end_chars = {'.', '?', '!', '\n'}

    # for chunk in completion:
    #     content = chunk[0]
    #     if content is not None:
    #         for char in content:
    #             sentence += char
    #             if char in sentence_end_chars:
    #                 sentence = sentence.strip()
    #                 if sentence and sentence not in sentences:
    #                     sentences.append(sentence)
    #                     speak(sentence)
    #                     print(f"Queued sentence: {sentence}")  # Logging queued sentence
    #                 sentence = ''
    # return sentences
    
    # st.markdown(md, unsafe_allow_html=True)


def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript