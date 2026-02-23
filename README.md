# 📊 AI Data Analyst Agent

An autonomous, pure Python data analysis application built with **Streamlit** and **LangChain**. 

This tool allows users to upload raw tabular data (CSV, Excel) or unstructured documents (PDF, DOCX) and interact with them using natural language. It writes and executes Python code in the background for data analysis, renders interactive drag-and-drop dashboards, and generates highly professional, manager-ready executive reports.

---

## ✨ Key Features

* **💬 Chat with Data (CSV & Excel):** Upload datasets and ask questions in plain English (e.g., *"What were the top 3 selling products in Q4?"*). The AI writes the Pandas code, executes it, and delivers the answer.
* **📈 Automated Visualizations:** Ask the AI to plot graphs and charts. It will generate them using Matplotlib/Seaborn and display them directly in the chat interface.
* **🖱️ Interactive Drag-and-Drop Dashboard:** Features a **PyGWalker** integration. Flip to the dashboard tab to turn your uploaded tabular data into a Tableau-style interface for manual, code-free visual exploration.
* **📚 Document Analysis (RAG Pipeline):** Upload PDFs or Word documents. The app chunks the text, creates vector embeddings using local HuggingFace models, and allows you to semantically search and ask questions about the document using Google Gemini.
* **📄 Executive Report Generation:** With one click, the app summarizes your entire analytical conversation and generates a formatted HTML web dashboard or a plain-text PDF executive summary for download.
* **🧠 Continuous Memory:** The chat interface remembers your previous questions during the session, allowing for deep-dive, contextual follow-up questions.

---

## 🛠️ Tech Stack

* **Frontend & Hosting:** Streamlit
* **LLM & Orchestration:** Google Gemini 2.5 Flash, LangChain, LangChain Experimental
* **Data Manipulation:** Pandas, OpenPyXL
* **Data Visualization:** PyGWalker, Matplotlib, Seaborn
* **RAG & Embeddings:** HuggingFace (`all-MiniLM-L6-v2`), FAISS, PyPDF, Docx2txt
* **Report Generation:** FPDF

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* A Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
Create and activate a virtual environment:

Bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
Install the dependencies:

Bash
pip install -r requirements.txt
Set up environment variables:
Create a .env file in the root directory of the project and add your Gemini API key:

Code snippet
GEMINI_API_KEY="your_actual_api_key_here"
Run the application:

Bash
streamlit run main.py
## 💡 Usage Guide
Upload a File: Use the sidebar to upload a .csv, .xlsx, .pdf, or .docx file.

Tabular Data (CSV/XLSX): * Navigate to the 💬 AI Chat Analyst tab to ask questions and generate AI charts.

Navigate to the 📈 PyGWalker Dashboard tab to manually drag and drop columns into custom charts.

Unstructured Data (PDF/DOCX): Ask questions about the document's content, and the RAG pipeline will extract the relevant context to answer.

Generate Reports: Once you have analyzed your data, open the left sidebar and click Generate Reports (HTML & PDF) to download a summary of your findings.

Clear Chat: Use the trash can icon in the sidebar to wipe the AI's memory before uploading a new, unrelated document.

## 🔒 Security Note
This application utilizes LangChain's allow_dangerous_code=True parameter to allow the Pandas agent to execute Python code dynamically. It is intended for local execution, portfolio demonstration, or containerized/sandboxed deployments.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


