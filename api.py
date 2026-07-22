from fastapi import FastAPI, Depends, HTTPException, Header
from contextlib import asynccontextmanager
from pydantic import BaseModel
from src.knowledge_agent import ask
from src.ingest import ingest_all_pdfs
from dotenv import load_dotenv
from src.database import get_recent, init_db
from src.limiter import limiter
from fastapi import Request
from fastapi import UploadFile, File
from src.ingest import ingest_pdf
import os
import shutil

load_dotenv()

api_key = os.getenv("API_KEY")

class UserQuery(BaseModel):
    question:str

def verify_api_key(x_api_key:str = Header()):
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail = "Invalid API key")
    
@asynccontextmanager
async def lifespan(app:FastAPI):
    init_db()
    ingest_all_pdfs()
    yield  

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
@app.get("/history")
@limiter.limit("5/minute")
def history(request:Request,status = Depends(verify_api_key)):
    return {"history": get_recent(limit = 10)}

@app.post("/ask")
@limiter.limit("5/minute")
def query(request:Request,body:UserQuery, status = Depends(verify_api_key)):
    answer = ask(body.question)
    return f"Response: {answer}"

@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_file(file:UploadFile = File(),status = Depends(verify_api_key)):
    temp_path = f"./data/pdfs/{file.filename}"
    with open(temp_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    ingest_pdf(temp_path)

    return f"Successfully ingested {file.filename}"

    
