import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import os
from langchain_core.documents import Document

client = chromadb.PersistentClient(path = "./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 40
)

def ingest_pdf(path:str):
    print(f"ingesting docs")
    
    
    existing = collection.get(where = {"source":path})

    if existing["ids"]:
        print(f"skipping{os.path.basename(path)}, already ingested")
        return
    print(f"Existing check result: {existing}")
    loader = PyPDFLoader(path)
    documents = loader.load()
    chunks = splitter.split_documents(documents)
    print(f"pages loaded: {len(documents)}")
    print(f"chunks created: {len(chunks)}")

    for i,chunk in enumerate(chunks):
        collection.add(
            documents = [chunk.page_content],
            ids=[f"pdf_{os.path.basename(path)}_{i}"],
            metadatas=[{
                "source":path,
                "type":"pdf",
                "page":str(chunk.metadata.get("page",0))
            }]
        )
    print(f"stored{len(chunks)} chunks from the pdf {os.path.basename(path)}")


def ingest_youtube(url:str):
    print(f"Ingesting YouTube:{url}")

    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        print("Invalid URL")
        return
    yapi = YouTubeTranscriptApi()
    transcript = yapi.fetch(video_id)

    segments = []
    for t in transcript:
        segments.append(t.text)
    full_text = " ".join(segments)

        

    chunks = splitter.split_text(full_text)

    for i,chunk in enumerate(chunks):
       
        collection.add(
            documents = [chunk],
            ids = [f"yt_{video_id}_{i}"],
            metadatas=[{
                "source":url,
                "type":"youtube",
                "video_id":video_id
            }]  
            )
    
    print(f"stored {len(chunks)} chunks from Youtube video")

def ingest_all_pdfs():
    pdf_folder = "./data/pdfs"
    os.makedirs(pdf_folder,exist_ok = True)
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            ingest_pdf(os.path.join(pdf_folder,filename))

if __name__ == "__main__":
    ingest_all_pdfs()

