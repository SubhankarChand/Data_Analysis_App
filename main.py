import streamlit as st
import tempfile
import os
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Files 📊")
    st.header("Ask your Files 📊")
    
    user_csv = st.file_uploader("Upload your file here", type="csv")
    
    if user_csv is not None:
        user_question = st.text_input("Ask a question about your file: ")
        
        # Save the Streamlit uploaded file to a temporary physical file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(user_csv.getvalue())
            tmp_file_path = tmp_file.name
        
        # Initialize the Gemini model 
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Create the agent using the updated experimental library
        agent = create_csv_agent(
            llm, 
            tmp_file_path, 
            verbose=True,
            allow_dangerous_code=True # Required by LangChain to execute pandas code
        )
        
        if user_question:
            with st.spinner("Analyzing..."):
                try:
                    # .run() is deprecated in newer versions, use .invoke()
                    response = agent.invoke(user_question)
                    st.write(f"**Your question was:** {user_question}")
                    st.write(f"**Answer:** {response['output']}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up the temporary file
                    os.remove(tmp_file_path)

if __name__ == "__main__":
    main()