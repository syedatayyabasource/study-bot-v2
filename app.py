import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Study Assistant")

# --- YEH NEW ADDITION HAI FRONTEND KE LIYE ---
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        # Yeh aapki index.html file ko link par load karega
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Index.html not found!</h1><p>Please make sure you uploaded index.html to your GitHub repo.</p>"

# AI Setup with error checking
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=api_key
)

# Study Bot Prompt Logic
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Study Assistant. Answer academic questions only. Be concise and educational."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

# MongoDB Memory setup
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

class ChatInput(BaseModel):
    session_id: str
    message: str  # Frontend 'message' bhej raha hai, isliye isay 'message' kar diya

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    try:
        if not data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        response = bot_with_history.invoke(
            {"question": data.message},
            config={"configurable": {"session_id": data.session_id}}
        )
        return {"response": response.content}
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
