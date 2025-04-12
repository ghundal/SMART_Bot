from typing import List, Optional

from fastapi import FastAPI

from pydantic import BaseModel

 

app = FastAPI()

 

# Define a Pydantic model for request/response body

class ChatMessage(BaseModel):
    content: str

 

# Path operation with type hints

@app.post("/chat/{chat_id}")

async def send_chat_message(chat_id: str, message: ChatMessage):

    return message.content

 

# Path parameter with type hint

@app.get("/items/{item_id}")

async def read_item(item_id: int):

    return {"item_id": item_id}

 

# Query parameter with type hint and default value

@app.get("/search/")

async def search_item(query: str, limit: int = 10):

    return {"query": query, "limit": limit}