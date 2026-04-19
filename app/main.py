"""
FastAPI web framework for the Coding Agent API
"""

import json
import sqlite3
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.models import GenerateApiKeyRequest, PromptRequest
from app.agent import run_agent_stream


# SQLite database
DB_FILE = "main.db"
CHAT_HASH_DIGITS = 12
API_KEY_HEADER = "api_key"

app = FastAPI(
    title="Coding Agent API",
    description="Simple AI Agent for coding assistance with streaming response and chat storage",
    version="1.0.0"
)

# Add CORS middleware - must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:3000",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Chat-ID"],
    max_age=600,
)


def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_hash TEXT UNIQUE,
            chat_name TEXT NOT NULL,
            context TEXT,
            all_chat TEXT NOT NULL
        )
    ''')
    cursor.execute("PRAGMA table_info(chats)")
    columns = {column[1] for column in cursor.fetchall()}
    if "chat_hash" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN chat_hash TEXT")

    cursor.execute("SELECT id FROM chats WHERE chat_hash IS NULL OR chat_hash = ''")
    rows_missing_hash = cursor.fetchall()
    for row in rows_missing_hash:
        cursor.execute("UPDATE chats SET chat_hash = ? WHERE id = ?", (generate_chat_hash(cursor), row[0]))

    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chats_chat_hash ON chats(chat_hash)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL UNIQUE,
            expiry_time TEXT NOT NULL,
            user_id INTEGER NOT NULL
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
    conn.commit()
    conn.close()


def generate_chat_hash(cursor) -> str:
    """Generate a frontend-safe unique numeric chat identifier."""
    lower_bound = 10 ** (CHAT_HASH_DIGITS - 1)
    upper_bound = (10 ** CHAT_HASH_DIGITS) - 1

    while True:
        chat_hash = str(secrets.randbelow(upper_bound - lower_bound + 1) + lower_bound)
        cursor.execute("SELECT 1 FROM chats WHERE chat_hash = ?", (chat_hash,))
        if cursor.fetchone() is None:
            return chat_hash


def parse_expiry_time(expiry_time: str) -> datetime:
    """Parse API key expiry timestamps stored in SQLite."""
    normalized_expiry = expiry_time.strip()
    if normalized_expiry.endswith("Z"):
        normalized_expiry = f"{normalized_expiry[:-1]}+00:00"

    try:
        parsed_expiry = datetime.fromisoformat(normalized_expiry)
    except ValueError:
        parsed_expiry = datetime.strptime(normalized_expiry, "%Y-%m-%d %H:%M:%S")

    if parsed_expiry.tzinfo is None:
        return parsed_expiry.replace(tzinfo=timezone.utc)
    return parsed_expiry.astimezone(timezone.utc)


async def require_valid_api_key(
    api_key: Optional[str] = Header(default=None, convert_underscores=False),
) -> int:
    """Validate the api_key header and return its user_id."""
    if not api_key:
        raise HTTPException(status_code=401, detail=f"Missing {API_KEY_HEADER} header")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, expiry_time FROM api_keys WHERE api_key = ?", (api_key,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_id, expiry_time = row
    try:
        expires_at = parse_expiry_time(expiry_time)
    except ValueError:
        raise HTTPException(status_code=500, detail="API key expiry time is invalid")

    if expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="API key has expired")

    return user_id


# Initialize database on startup
init_db()


@app.post("/generate_api_key")
async def generate_api_key(request: GenerateApiKeyRequest):
    """Generate and store a new API key for a user."""
    if request.expiry_days is not None and request.expiry_days <= 0:
        raise HTTPException(status_code=400, detail="expiry_days must be greater than 0")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    new_api_key = secrets.token_urlsafe(32)
    expiry_days = request.expiry_days or 2
    expiry_time = (datetime.now(timezone.utc) + timedelta(days=expiry_days)).isoformat()
    cursor.execute(
        "INSERT INTO api_keys (api_key, expiry_time, user_id) VALUES (?, ?, ?)",
        (new_api_key, expiry_time, request.user_id),
    )
    conn.commit()
    api_key_id = cursor.lastrowid
    conn.close()
    return {
        "id": api_key_id,
        "api_key": new_api_key,
        "expiry_time": expiry_time,
        "user_id": request.user_id,
    }


# @app.post("/api/generate")
# async def generate_response(request: PromptRequest, user_id: int = Depends(require_valid_api_key)):
#     """Generate an agent response for callers with a valid API key."""
#     def generate():
#         for token in run_agent_stream(request.prompt):
#             yield token

#     headers = {
#         "Cache-Control": "no-cache, no-transform",
#         "X-Accel-Buffering": "no",
#         "X-User-ID": str(user_id),
#     }
#     return StreamingResponse(generate(), media_type="text/plain; charset=utf-8", headers=headers)


@app.post("/api/generate")
async def query_agent(request: PromptRequest):
    """Handle agent queries - create new chat or continue existing chat"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Case 1: No chat_id provided - Create new chat
    if request.chat_id is None:
        chat_hash = generate_chat_hash(cursor)
        chat_name = re.sub(r'[^\w\s]', '', request.prompt)[:50].strip() or "untitled"
        context_list = []  # Start with empty context
        messages = []
        
        # Insert new chat with empty context
        cursor.execute("INSERT INTO chats (chat_hash, chat_name, context, all_chat) VALUES (?, ?, ?, ?)", 
                      (chat_hash, chat_name, json.dumps(context_list), json.dumps(messages)))
        conn.commit()
        db_chat_id = cursor.lastrowid
        conn.close()
    
    # Case 2: chat_id provided - Update existing chat
    else:
        chat_hash = str(request.chat_id)
        cursor.execute("SELECT id, context, all_chat FROM chats WHERE chat_hash = ?", (chat_hash,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_hash} not found")
        
        db_chat_id = row[0]
        context_list = json.loads(row[1]) if row[1] else []
        messages = json.loads(row[2]) if row[2] else []
        conn.close()
    
    # Define streaming generator - yields tokens immediately as they arrive.
    # Keep this as a normal generator so Starlette can stream it from a worker
    # thread instead of blocking the event loop while requests waits on Ollama.
    def generate():
        response_parts = []
        try:
            # Stream tokens from Ollama API immediately
            for token in run_agent_stream(request.prompt, context_list):
                response_parts.append(token)
                yield token  # Yield immediately without buffering
        finally:
            # After streaming completes, update the chat database
            full_response = "".join(response_parts)
            messages.append({"user": request.prompt})
            messages.append({"agent": full_response})
            
            # Add response to context for next queries
            response_num = len([item for item in context_list if "Response" in list(item.keys())[0] if item]) + 1
            context_list.append({f"Response{response_num}": full_response})
            
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("UPDATE chats SET context = ?, all_chat = ? WHERE id = ?", 
                          (json.dumps(context_list), json.dumps(messages), db_chat_id))
            conn.commit()
            conn.close()
    
    # Return streaming response with chatID in headers
    headers = {
        "X-Chat-ID": chat_hash,
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/chats")
async def get_chats():
    """Get all chats"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_hash, chat_name, context, all_chat FROM chats ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    chats = []
    for row in rows:
        messages = json.loads(row[3]) if row[3] else []
        context_list = json.loads(row[2]) if row[2] else []
        chats.append({
            "id": row[0], 
            "chat_name": row[1], 
            "context": context_list, 
            "messages": messages
        })
    return {"chats": chats}


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: int):
    """Get a specific chat with all messages and context"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_hash, chat_name, context, all_chat FROM chats WHERE chat_hash = ?", (str(chat_id),))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
    
    messages = json.loads(row[3]) if row[3] else []
    context_list = json.loads(row[2]) if row[2] else []
    return {
        "id": row[0],
        "chat_name": row[1],
        "context": context_list,
        "messages": messages
    }

def check_api_key_validity(api_key: str) -> bool:
    """Check if the provided API key is valid and not expired."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT expiry_time FROM api_keys WHERE api_key = ?", (api_key,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    expiry_time = row[0]
    try:
        expires_at = parse_expiry_time(expiry_time)
    except ValueError:
        return False

    return expires_at > datetime.now(timezone.utc)


if __name__ == "__main__":
    import uvicorn
    print("Starting the Coding Agent API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
