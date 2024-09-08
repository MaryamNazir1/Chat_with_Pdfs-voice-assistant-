import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from utils import autoplay_audio, transcribe_audio, generate_speech
import google.generativeai as genai  # Ensure this import is present
st.set_page_config(page_title="Interact with Your Documents", page_icon="📝")
def load_env():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = []  # Clear all messages
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How may I assist you today?"})  # Add initial message
    # Clear the audio input history
    st.session_state.audio_query = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context."
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response['output_text']  # Access the output_text directly
def main():
    load_env()
    initialize_session_state()

    # Initialize floating features for the interface
    float_init()

    st.title("Interact with Your Documents 📝")  # Changed the main heading

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Options:")  # Changed the menu word
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Process PDF files
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)  # Chunk the text
                    get_vector_store(text_chunks)  # Create vector store
                    st.success("Done")
                else:
                    st.error("No text extracted from the PDFs.")

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Create a container for the footer (mic and input box)
    footer_container = st.container()

    with footer_container:
        # Create two columns: one for the input box and one for the audio recorder
        col1, col2 = st.columns([9, 1])  # Adjust ratio as needed
        
        with col1:
            # Input box for text input
            text_input = st.chat_input('')
        with col2:
            # Audio recorder for voice input
            audio_bytes = audio_recorder(text="", icon_size="2x", recording_color="#7C0A02", neutral_color="#FFFFFF")

    # Float the footer container at the bottom of the screen
    footer_container.float("bottom:20px;")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Handle audio input (only process new audio)
    if audio_bytes and st.session_state.get('processed_audio_query') != audio_bytes:
        with st.chat_message("user"):
            with st.spinner("Transcribing..."):
                # Write the audio bytes to a temporary file
                webm_file_path = "temp_audio.wav"
                with open(webm_file_path, "wb") as f:
                    f.write(audio_bytes)

                # Convert the audio to text
                transcript = transcribe_audio(webm_file_path)
                os.remove(webm_file_path)
                if transcript:
                    st.session_state.messages.append({"role": "user", "content": transcript})
                    st.session_state.audio_query = transcript  # Store the processed audio query
                    st.session_state.processed_audio_query = audio_bytes  # Mark audio as processed
                    st.write(transcript)

    # Check if there is any user input (text or voice)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_question = st.session_state.messages[-1]["content"]  # Get the last user input
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                final_response = user_input(user_question)  # Pass the user's question
                st.write(final_response)  # Display the full response
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            with st.spinner("Generating audio response..."):    
                audio_file = generate_speech(final_response)
                autoplay_audio(audio_file)  # Play the audio response
                os.remove(audio_file)  # Clean up the audio file after playing

    # Check if there is any text input
    if text_input:
        with st.chat_message("user"):
            st.write(text_input)  # Display the user's text input
            st.session_state.messages.append({"role": "user", "content": text_input})  

        # Handle text input search
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                final_response = user_input(text_input)  # Pass the text input as the user's question
                st.write(final_response)  # Display the full response
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            with st.spinner("Generating audio response..."):    
                audio_file = generate_speech(final_response)
                autoplay_audio(audio_file)  # Play the audio response
                os.remove(audio_file)  # Clean up the audio file after playing

if __name__ == "__main__":
    main()
