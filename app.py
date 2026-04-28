import sys
from pathlib import Path

# Compatibility fix for Python versions older than 3.10.
# Some Google client libraries expect importlib.metadata.packages_distributions to exist.
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata as importlib_metadata

if not hasattr(importlib_metadata, "packages_distributions"):
    try:
        import importlib_metadata as backport
        importlib_metadata.packages_distributions = backport.packages_distributions
    except ImportError:
        pass

import os
import time
import re
import pandas as pd
import gradio as gr
import chromadb
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# --- GEMINI & LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# ==========================================
# CONFIGURATION
# ==========================================
dotenv_path = Path(__file__).resolve().parent / ".env"
print(f"Loading .env from {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)
print(f"GOOGLE_API_KEY loaded: {'yes' if os.getenv('GOOGLE_API_KEY') else 'no'}")

INPUT_CSV = "Constitution Of India.csv"
PROCESSED_CSV = "multilingual_constitution.csv"
CHROMA_DB_DIR = "chroma_db_gemini"
COLLECTION_NAME = "constitution_store_gemini"

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please set it in .env file or deployment secrets.")

# --- CUSTOM CSS ---
# This CSS handles the dark-mode aesthetic and text alignment
custom_css = """
#main-container { background-color: #0f172a; padding: 20px; }
#title-text { text-align: center; color: #38bdf8; margin-bottom: 20px; font-size: 2.5em; font-weight: bold; }
.gradio-container { font-family: 'Inter', sans-serif !important; }
#lang-selector { border-radius: 10px; border: 1px solid #334155; }
"""

# --- THEME SETUP ---
# Creating a soft sky-blue theme for a premium feel
custom_theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    block_border_width="1px",
)

# ==========================================
# STEP 1: DATA PROCESSING
# ==========================================
def process_data():
    if not os.path.exists(INPUT_CSV):
        dummy_data = {"Articles": ["1. Name and territory of the Union.", "21. Protection of life and personal liberty."]}
        pd.DataFrame(dummy_data).to_csv(INPUT_CSV, index=False)
    
    if os.path.exists(PROCESSED_CSV):
        return pd.read_csv(PROCESSED_CSV)

    df = pd.read_csv(INPUT_CSV)
    processed_rows = []
    languages = {'hi': 'Hindi', 'ta': 'Tamil', 'mr': 'Marathi'}

    for index, row in df.iterrows():
        original_text = str(row.get("Articles", row.iloc[0])).strip()
        match = re.match(r'^"?(\d+[A-Z]?)\.', original_text)
        article_id = match.group(1) if match else f"Row_{index+1}"
        processed_rows.append({"article_id": article_id, "language": "English", "content": original_text})

        for lang_code, lang_name in languages.items():
            try:
                translated = GoogleTranslator(source='auto', target=lang_code).translate(original_text)
                processed_rows.append({"article_id": article_id, "language": lang_name, "content": translated})
            except Exception:
                continue 

    new_df = pd.DataFrame(processed_rows)
    new_df.to_csv(PROCESSED_CSV, index=False)
    return new_df

# ==========================================
# STEP 2: INITIALIZE GEMINI AI
# ==========================================
def initialize_app():
    if not os.path.exists(CHROMA_DB_DIR):
        df = process_data()
    else:
        df = None

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    
    if vectorstore._collection.count() == 0 and df is not None:
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=str(row["content"]), 
                metadata={"article_id": str(row["article_id"]), "language": str(row["language"])}
            )
            documents.append(doc)
            
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            vectorstore.add_documents(documents[i:i+batch_size])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=2)
    return vectorstore, llm

# Initialize global variables
vectorstore, llm = None, None
try:
    vectorstore, llm = initialize_app()
except Exception as e:
    print(f"INITIALIZATION ERROR: {e}")

# ==========================================
# STEP 3: CHAT LOGIC
# ==========================================
def chat_logic(message, history, language_selection):
    if not vectorstore or not llm:
        return "System Error: Database or API not initialized."
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "filter": {"language": language_selection}}
        )
        system_prompt = (
            "You are an expert legal assistant specialized in the Constitution of India. "
            "Use the context to answer. Reply in {language}. "
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": message, "language": language_selection})
        return response["answer"]
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# STEP 4: UPGRADED UI SETUP
# ==========================================
with gr.Blocks(title="Constitution AI", theme=custom_theme, css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown("# Constitution AI 🇮🇳", elem_id="title-text")
        
        with gr.Row():
            # Use an accordion to hide settings and focus on chat
            with gr.Accordion("Settings & Language Selection", open=True):
                lang_dropdown = gr.Dropdown(
                    choices=["English", "Hindi", "Tamil", "Marathi"], 
                    value="English", 
                    label="Interaction Language",
                    elem_id="lang-selector"
                )
                gr.Markdown("Select your preferred language for the legal search and AI response.")

        # Chat interface with added helpful examples
        gr.ChatInterface(
            fn=chat_logic, 
            additional_inputs=[lang_dropdown], 
            type="messages",
            examples=[["What is Article 21?"], ["Explain the Right to Equality"], ["What is the Union Territory?"]]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
