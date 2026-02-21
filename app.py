from mangum import Mangum
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Study Assistant")

# --- CORS SETTINGS (Frontend connection ke liye zaroori) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Index.html not found!</h1>"

# AI Setup
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Study Assistant. Answer academic questions only. Be concise and educational."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

def get_memory(session_id: str):
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI not found")
    
    return MongoDBChatMessageHistory(
        connection_string=uri, 
        session_id=session_id,
        database_name="star_square", 
        collection_name="study_bot_history"
    )

bot_with_history = RunnableWithMessageHistory(
    chain, 
    get_memory, 
    input_messages_key="question", 
    history_messages_key="history"
)

# --- FRONTEND SE MATCH KARNE KE LIYE YAHAN 'question' KIYA HAI ---
class ChatInput(BaseModel):
    session_id: str
    question: str  # Frontend 'question' bhej raha hai

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    try:
        if not data.question.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        response = bot_with_history.invoke(
            {"question": data.question},
            config={"configurable": {"session_id": data.session_id}}
        )
        return {"response": response.content}
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
handler = Mangum(app)
