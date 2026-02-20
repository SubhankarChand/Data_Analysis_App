import streamlit as st
import tempfile
import os
import pandas as pd
from fpdf import FPDF
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# PyGWalker for the interactive dashboard
from pygwalker.api.streamlit import StreamlitRenderer

# RAG imports from your existing file
from rag_utils import extract_text_from_file, get_text_chunks, get_vector_store, get_conversational_chain

# Cache the PyGWalker renderer so it doesn't reload on every chat message
@st.cache_resource
def get_pyg_renderer(dataframe):
    return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="rw")

# Helper function to create PDF bytes
def create_pdf(text_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    # Clean text to prevent character encoding errors in FPDF
    safe_text = text_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=safe_text)
    return pdf.output(dest='S').encode('latin-1')

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Files 📊", page_icon="📊", layout="wide")
    st.header("Ask your Files 📊")
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Controls & Reports")
        
        # 1. Clear Chat Button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        st.subheader("Generate Executive Report")
        
        if st.button("📄 Generate Reports (HTML & PDF)"):
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                with st.spinner("Writing professional reports..."):
                    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    report_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                    
                    # 1. Generate HTML Report
                    html_prompt = f"""
                    Act as a Senior Data Analyst. Write a highly professional executive summary based on this conversation.
                    CRITICAL: Output ONLY valid HTML with inline CSS (Arial font, clean spacing). Do NOT use markdown.
                    Include: Executive Summary, Key Findings, Visualizations Discussed, and Recommendations.
                    Conversation: {history_text}
                    """
                    report_html = report_llm.invoke(html_prompt).content
                    report_html = report_html.replace("```html", "").replace("```", "").strip()
                    
                    # 2. Generate Plain Text Report for PDF
                    pdf_prompt = f"""
                    Act as a Senior Data Analyst. Write a professional executive summary based on this conversation.
                    CRITICAL: Output ONLY plain text. Do NOT use markdown, asterisks, bolding, or HTML.
                    Include: Executive Summary, Key Findings, Visualizations Discussed, and Recommendations.
                    Conversation: {history_text}
                    """
                    report_text = report_llm.invoke(pdf_prompt).content
                    pdf_bytes = create_pdf(report_text)
                    
                    st.success("Reports generated successfully!")
                    
                    # HTML Download Button
                    st.download_button(
                        label="⬇️ Download Report (HTML)",
                        data=report_html,
                        file_name="Executive_Report.html",
                        mime="text/html"
                    )
                    
                    # PDF Download Button
                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="Executive_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("Chat is empty! Ask some questions first.")
    # ------------------------

    # FIXED FILE UPLOADER: Now explicitly accepts all three formats
    uploaded_file = st.file_uploader("Upload your file here (CSV, xlsx, PDF, DOCX)", type=["csv", "xlsx", "pdf", "docx"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ==========================================
        # ROUTE 1: CSV & EXCEL DATA ANALYSIS & PYGWALKER DASHBOARD
        # ==========================================
        if file_extension in ["csv", "xlsx"]:
            # 1. Read the file into a Pandas DataFrame based on its extension
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file) # Requires openpyxl installed
            
            # The tabs that separate Chat from the PyGWalker Dashboard
            tab1, tab2 = st.tabs(["💬 AI Chat Analyst", "📈 PyGWalker Dashboard"])
            
            with tab1:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if user_question := st.chat_input("Ask a question about your data:"):
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.markdown(user_question)
                        
                    with st.spinner("Analyzing data..."):
                        try:
                            # 2. Save the dataframe to a temporary CSV so the agent can read it
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                                df.to_csv(tmp_file.name, index=False)
                                tmp_file_path = tmp_file.name
                                
                            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                            agent = create_csv_agent(llm, tmp_file_path, verbose=True, allow_dangerous_code=True)
                            
                            graph_instruction = "\nIf asked for a graph, save it as 'temp_chart.png'. Do not use plt.show()."
                            response = agent.invoke(user_question + graph_instruction)
                            answer = response['output']
                            
                            with st.chat_message("assistant"):
                                st.markdown(answer)
                                if os.path.exists("temp_chart.png"):
                                    st.image("temp_chart.png")
                                    os.remove("temp_chart.png")
                                    
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            os.remove(tmp_file_path)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            # The PyGWalker Interactive Drag-and-Drop Interface
            with tab2:
                st.subheader("Interactive Data Visualization")
                st.info("Drag and drop your columns from the left panel onto the X and Y axes to build custom charts.")
                renderer = get_pyg_renderer(df)
                renderer.explorer()
                
        # ==========================================
        # ROUTE 2: PDF & DOCX ANALYSIS
        # ==========================================
        elif file_extension in ["pdf", "docx"]:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if user_question := st.chat_input(f"Ask a question about your {file_extension.upper()} file:"):
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                    
                with st.spinner("Reading document..."):
                    try:
                        raw_text = extract_text_from_file(uploaded_file)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        docs = vector_store.similarity_search(user_question)
                        
                        chain = get_conversational_chain()
                        response = chain.invoke(
                            {"input_documents": docs, "question": user_question},
                            return_only_outputs=True
                        )
                        
                        answer = response['output_text']
                        
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()