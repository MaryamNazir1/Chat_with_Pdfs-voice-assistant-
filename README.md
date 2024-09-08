# Document Interaction and Voice Assistant

This Streamlit app allows users to interact with their PDF documents through text and voice. Users can upload PDF files, ask questions based on the document content, and receive text and audio responses using Google Generative AI for question answering and Deepgram for text-to-speech generation.

## Features
- **PDF Upload and Processing:** Upload multiple PDF files, extract text, and split it into manageable chunks for querying.
- **Text and Voice Input:** Users can either type questions or record their voice. The app transcribes voice input into text for querying.
- **Conversational Chain:** Uses Langchain's conversational chain to answer questions based on the provided context.
- **Voice Response:** The app generates an audio response for the user’s question using Deepgram's text-to-speech API.
- **Floating UI Components:** The app has a floating UI design that keeps the input and microphone buttons at the bottom.

## Setup and Installation

### Requirements

Ensure that you have Python 3.8+ installed on your machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/maryamnazir1/document-voice-assistant.git
   cd document-voice-assistant
   ```

2. **Install dependencies**:
   Install the required Python libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in the root directory with the following content:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   DG_API_KEY=your_deepgram_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

### Running the App

To run the app, use the following command:
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501` by default.

## File Structure

```
.
├── app.py              # Main application script
├── utils.py            # Utility functions for audio and transcription
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
└── README.md           # Project documentation
```

### `app.py`
This is the main script that handles the Streamlit interface, PDF processing, question-answering logic, and voice interaction.

### `utils.py`
Contains helper functions for generating speech, transcribing audio, and handling file-based audio playback.

## Features Overview

### 1. PDF Upload
- Users can upload multiple PDF files via the sidebar.
- The app processes and extracts text from these PDFs.

### 2. Conversational Chain
- The app uses Langchain's conversational chain, powered by Google Generative AI, to answer questions based on the provided PDF content.

### 3. Voice and Text Input
- Users can input queries through text or record audio using the microphone button.
- The audio is transcribed using Groq’s transcription API, and responses are provided both in text and audio form.

### 4. Audio Response
- The app generates audio responses using Deepgram’s text-to-speech API and plays the response directly in the browser.

## Key Technologies

- **Streamlit:** For creating the web interface.
- **Google Generative AI (Gemini):** Used for generating conversational AI responses.
- **Deepgram:** For generating speech from text.
- **Groq:** For audio transcription and language models.
- **Langchain:** For managing the conversational logic and vector storage using FAISS.

## Future Enhancements

- Add support for additional document formats.
- Improve the accuracy of the voice-to-text transcription.
- Include more language models for diverse question answering.
