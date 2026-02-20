import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
# 1. Extract Text from the Uploaded File
def extract_text_from_file(uploaded_file):
    # Create a temporary file to allow LangChain loaders to read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
        elif uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_file_path)
            pages = loader.load()
        else:
            raise ValueError("Unsupported file format")
            
        # Combine all pages into one large string of text
        text = "".join([page.page_content for page in pages])
        return text
    finally:
        os.remove(tmp_file_path) # Always clean up

# 2. Split Text into Chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 3. Create Embeddings and Store in FAISS
def get_vector_store(text_chunks):
    # Using Gemini's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Create the local vector database
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# 4. Setup the QA Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide the wrong answer.
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # load_qa_chain handles passing the retrieved documents to the LLM
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain