import streamlit as st
import tempfile
import os
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# PyGWalker imports
from pygwalker.api.streamlit import StreamlitRenderer

# RAG imports (ensure your rag_utils.py is in the same folder)
from rag_utils import extract_text_from_file, get_text_chunks, get_vector_store, get_conversational_chain

# Cache the PyGWalker renderer to prevent memory issues
@st.cache_resource
def get_pyg_renderer(dataframe):
    # This will save any drag-and-drop charts you make into a local config file
    return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="rw")

def main():
    load_dotenv()
    # Changed layout to "wide" to give the dashboard more room
    st.set_page_config(page_title="Ask your Files 📊", page_icon="📊", layout="wide")
    st.header("Ask your Files 📊")
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Controls")
        
        # 1. Clear Chat Button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun() # Refreshes the app instantly
            
        # 2. Generate Final Report Button
        if st.button("📄 Generate Final Report"):
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                with st.spinner("Writing professional report..."):
                    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    
                    report_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                    
                    # Advanced prompt for a PowerBI-style manager report
                    prompt = f"""
                    Act as a Senior Data Analyst. Using the conversation history below, write a highly professional, 
                    comprehensive markdown report suitable for a Manager or Executive. Structure it like a Power BI 
                    executive summary. Include:
                    1. Executive Summary
                    2. Key Findings & Insights (Use bullet points)
                    3. Visualizations Created (Mention any charts discussed)
                    4. Data Quality & Limitations
                    5. Actionable Recommendations
                    
                    Conversation History:
                    {history_text}
                    """
                    
                    report = report_llm.invoke(prompt).content
                    
                    st.download_button(
                        label="⬇️ Download Executive Report",
                        data=report,
                        file_name="Executive_Analysis_Report.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Chat is empty! Ask some questions first.")
    # ------------------------

    # Allow multiple file types
    uploaded_file = st.file_uploader("Upload your file here (CSV, PDF, DOCX)", type=["csv", "pdf", "docx"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ==========================================
        # ROUTE 1: CSV DATA ANALYSIS & DASHBOARD
        # ==========================================
        if file_extension == "csv":
            # Read into pandas dataframe for PyGWalker
            df = pd.read_csv(uploaded_file)
            
            # Create tabs for switching between Chat and Dashboard
            tab1, tab2 = st.tabs(["💬 AI Chat Analyst", "📈 Interactive Dashboard"])
            
            with tab1:
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat UI
                if user_question := st.chat_input("Ask a question about your CSV file:"):
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.markdown(user_question)
                        
                    with st.spinner("Analyzing data..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                                
                            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                            agent = create_csv_agent(llm, tmp_file_path, verbose=True, allow_dangerous_code=True)
                            
                            graphing_instruction = """
                            If you are asked to create a graph, chart, or plot, you must save it as 'temp_chart.png'. 
                            Do not use plt.show(). Use matplotlib or seaborn.
                            """
                            response = agent.invoke(user_question + graphing_instruction)
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

            with tab2:
                st.subheader("Drag and Drop Visualization")
                # Render PyGWalker UI
                renderer = get_pyg_renderer(df)
                renderer.explorer()

        # ==========================================
        # ROUTE 2: PDF & DOCX ANALYSIS
        # ==========================================
        elif file_extension in ["pdf", "docx"]:
            # Display chat history
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