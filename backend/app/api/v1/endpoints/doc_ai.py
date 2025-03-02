from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.api.v1.dependencies import verify_api_key  # Import API key protection
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import io
import openai
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
from langchain_community.chat_message_histories import ChatMessageHistory


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# ✅ Initialize OpenAI LangChain Client
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ FAISS Vector Store for Document Retrieval
vector_store = None

# ✅ Store Extracted Text
extracted_text = ""

# ✅ Define Chat State
class ChatState:
    def __init__(self):
        self.memory = ChatMessageHistory()
        self.vector_store = None

chat_state = ChatState()

# ✅ Define Graph for Processing Queries
graph = StateGraph(ChatState)

def process_user_query(state, query):
    if state.vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")

    retriever = state.vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)

    response_text = "\n".join([doc.page_content for doc in docs])
    state.memory.add_user_message(query)
    state.memory.add_ai_message(response_text)

    return {"response": response_text}

graph.add_node("process_query", process_user_query)
graph.set_entry_point("process_query")
conversation_graph = graph.compile()

 # 📌 1. Upload & Process Documents (Protected with API Key)
@router.post("/upload-doc")
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)  # Require valid API key
):
    """
    Uploads and processes a document (PDF) for retrieval-based Q&A.
    """
    global extracted_text, chat_state

    try:
        # Read file into memory
        file_content = await file.read()
        file_path = f"temp_{file.filename}"

        # Save file temporarily
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Load document using PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Extract text from document
        extracted_text = "\n".join([page.page_content for page in pages])

        # Generate embeddings and store in FAISS
        chat_state.vector_store = FAISS.from_documents(pages, embeddings)

        # Cleanup file
        os.remove(file_path)

        return {"message": "Document processed successfully!", "extracted_text": extracted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 📌 2. Get Extracted Text
@router.get("/get-text")
async def get_extracted_text():
    """
    Returns the extracted text from the uploaded document.
    """
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")

    return {"extracted_text": extracted_text}


# 📌 3. Send Extracted Text to AI
class ProcessRequest(BaseModel):
    text: str

@router.post("/send-to-ai")
async def send_to_ai():
    """
    Sends extracted text to an external AI service for further processing.
    """
    global extracted_text

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")

    try:
        # Send extracted text to external AI service
        ai_api_url = "http://localhost:8001/process-text"  # Update with the correct AI API URL

        response = requests.post(ai_api_url, json={"text": extracted_text})

        if response.status_code == 200:
            return {"message": "Text sent to AI successfully!", "ai_response": response.json()}
        else:
            return {"error": f"Failed to process text in AI: {response.text}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 📌 4. Ask Questions About Uploaded Documents
class QueryRequest(BaseModel):
    query: str

@router.post("/query-doc")
async def query_document(request: QueryRequest):
    """
    Query the uploaded document using LangGraph-based conversational AI.
    """
    global conversation_graph

    try:
        response = conversation_graph.invoke({"query": request.query})
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
