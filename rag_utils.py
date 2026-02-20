import streamlit as st
import tempfile
import os
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Import the RAG functions from your rag_utils.py file
from rag_utils import extract_text_from_file, get_text_chunks, get_vector_store, get_conversational_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Files 📊", page_icon="📊")
    st.header("Ask your Files 📊")
    
    # 1. Update uploader to accept multiple file types
    uploaded_file = st.file_uploader("Upload your file here (CSV, PDF, DOCX)", type=["csv", "pdf", "docx"])
    
    if uploaded_file is not None:
        # Determine the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        user_question = st.text_input(f"Ask a question about your {file_extension.upper()} file: ")
        
        if user_question:
            with st.spinner("Analyzing..."):
                try:
                    # ==========================================
                    # ROUTE 1: CSV DATA ANALYSIS (Using Agent)
                    # ==========================================
                    if file_extension == "csv":
                        # Save Streamlit file to a temporary physical file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                        
                        agent = create_csv_agent(
                            llm, 
                            tmp_file_path, 
                            verbose=True,
                            allow_dangerous_code=True 
                        )
                        
                        response = agent.invoke(user_question)
                        st.write(f"**Answer:** {response['output']}")
                        
                        # Clean up
                        os.remove(tmp_file_path)

                    # ==========================================
                    # ROUTE 2: PDF & DOCX ANALYSIS (Using RAG)
                    # ==========================================
                    elif file_extension in ["pdf", "docx"]:
                        # 1. Extract text using the function from rag_utils
                        raw_text = extract_text_from_file(uploaded_file)
                        
                        # 2. Chunk text
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create Vector Store
                        vector_store = get_vector_store(text_chunks)
                        
                        # 4. Search for relevant information
                        docs = vector_store.similarity_search(user_question)
                        
                        # 5. Pass to the QA Chain
                        chain = get_conversational_chain()
                        response = chain.invoke(
                            {"input_documents": docs, "question": user_question},
                            return_only_outputs=True
                        )
                        
                        st.write(f"**Answer:** {response['output_text']}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()