import streamlit as st
import os
from back import llm_w_stream, speak, speech_to_text
from streamlit_float import *
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder


# Float feature initialization
float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Welcome to our laptop store, how may I help you?"}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi! Welcome to our laptop store, how may I help you?")
        ]

initialize_session_state()

st.title("OpenAI Conversational Chatbot 🤖")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=False,
        callback=None,
        args=(),
        kwargs={},
        key=None
    )
    audio_bytes = audio["bytes"] if audio is not None else None

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})

            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        question = st.session_state.messages[-1]["content"]
        with st.spinner("Generating🤔..."):
            final_response = llm_w_stream(question, st.session_state.chat_history)
        with st.spinner("Generating audio response..."):
            speak(final_response)
        #     audio_file = text_to_speech(final_response)
        #     autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=final_response))
        # os.remove(audio_file)

# Float the footer container and provide CSS to target it with
footer_container.float("bottom: 0rem;")