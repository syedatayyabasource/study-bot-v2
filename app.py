import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Study Assistant")

# AI Setup with error checking
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY not found!")

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

# MongoDB Memory setup with validation
def get_memory(session_id: str):
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI not found in environment variables")
    
    # Cleaning URI if it has trailing slashes or DB names that cause conflicts
    clean_uri = uri.split('?')[0].rstrip('/')
    if not clean_uri.endswith(".net"):
        # If there's a DB name in URI, we strip it to avoid conflict with database_name param
        base_uri = clean_uri.split('.net')[0] + ".net/"
        options = uri.split('.net/')[1] if '.net/' in uri else ""
        uri = base_uri + "?" + options

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
    question: str

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    try:
        # Check if input is empty
        if not data.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        response = bot_with_history.invoke(
            {"question": data.question},
            config={"configurable": {"session_id": data.session_id}}
        )
        return {"response": response.content}
    
    except Exception as e:
        # Ye terminal mein exact error dikhayega
        print(f"ERROR OCCURRED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def home():
    return {
        "status": "Study Bot is Live!",
        "mongo_status": "Configured" if os.getenv("MONGO_URI") else "Missing",
        "groq_status": "Configured" if os.getenv("GROQ_API_KEY") else "Missing"
    }