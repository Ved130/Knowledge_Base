from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import chromadb
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from src.database import save_convo, get_recent, init_db
from src.latency import stage, set_cache_status
import redis
import json
import os


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")



redis_url = os.getenv("REDIS_URL")
llm = ChatGroq(model="openai/gpt-oss-20b",api_key= os.getenv("GROQ_API_KEY"), temperature=0,max_retries=0)
cache = redis.from_url(redis_url,decode_responses=True)
CACHE_PREFIX = "kb:"
CACHE_TTL = 3600 #cache time to live
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    task: str
    sources: List[str]
    history: List[dict]
    
    

def route_decision(state: AgentState) -> str:
    question = state["question"].lower()

    if any(word in question for word in ["summarise", "summarize", "summary", "overview"]):
        return "summarise"
    elif any(word in question for word in ["study question", "quiz", "test me", "generate questions", "practice"]):
        return "study_questions"
    else:
        return "qa"

def router(state: AgentState) -> AgentState:
    return state

def retrieve(state: AgentState) -> AgentState:
    with stage("retrieval"):
        results = collection.query(
            query_texts=[state["question"]],
            n_results=5
        )
        sources = sorted({os.path.basename(m.get("source", "")) for m in results["metadatas"][0]})
    print(f"retrieved {len(results['documents'][0])} chunks from: {sorted(set(sources))}")
    context = "\n".join(results["documents"][0])
    print("RETRIEVE sources:", sources)
    return {"context": context,"sources":sources}

def generate(state: AgentState) -> AgentState:
    task = route_decision(state)
    history = ""

    for turn in state["history"]:
        history += f"User:{turn['question']}\nAssistant:{turn['answer']}\n\n"

    if task == "summarise":
        template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a helpful assistant. Summarise the following content clearly and concisely.
Focus on the key concepts and main ideas.

Content:
{context}

User request: {question}

Summary:"""
        )
        prompt = template.format(context=state["context"], question=state["question"])

    elif task == "study_questions":
        template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful study assistant. Generate clear study questions based on the content below.

Rules:
- Use ONLY the context below. If the answer is not there, reply exactly: "I don't have that information."
- Answer the question directly in 1-4 sentences. Use a short bullet list only if listing several items.
- Copy names, numbers, dates and contact details exactly as written in the context.
- Do not overstate results; keep qualifiers like "though", "competitive with", "remained stronger".
- Plain Markdown only: no tables, no HTML tags.
- Each context chunk starts with [Source: ...] naming its document.

Content:
{context}

User request: {question}

Study Questions:"""
        )
        prompt = template.format(context=state["context"], question=state["question"])

    else:
        # Default Q&A
        template = PromptTemplate(
            input_variables=["context", "history", "question"],
            template="""
You are a helpful assistant. Answer the question using only the context provided.
If the answer is not in the context say "I don't have that information."

Rules:
- Use ONLY the context below. If the answer is not there, reply exactly: "I don't have that information."
- Answer the question directly in 1-4 sentences. Use a short bullet list only if listing several items.
- Copy names, numbers, dates and contact details exactly as written in the context.
- Do not overstate results; keep qualifiers like "though", "competitive with", "remained stronger".
- Plain Markdown only: no tables, no HTML tags.
- Each context chunk starts with [Source: ...] naming its document.

Context:
{context}

Previous Conversation:
{history}

Question: {question}

Answer:"""
        )
        prompt = template.format(
            context=state["context"],
            history=history,
            question=state["question"]
        )
    with stage("llm"):
        response = llm.invoke(prompt)
    return {"answer": response.content}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "qa": "retrieve",
            "summarise": "retrieve",
            "study_questions": "retrieve"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

conversation_history = []
def clear_cache()->int:
    "To clear cache called after ingestion to prevent stale answers"
    try:
        keys = list(cache.scan_iter(match=f"{CACHE_PREFIX}*"))
        if keys:
            cache.delete(*keys)

        return len(keys)
    except redis.exceptions.RedisError as e:
        print(f"Cache clear failed{e}")
def ask(question: str) -> str:
    key = CACHE_PREFIX+question.lower().strip()
    cached = None
    with stage("cache_lookup"):
        try:
            cached = cache.get(key)
        except redis.exceptions.RedisError as e:
            print(f"Cache read failed{e}")
    set_cache_status(hit=cached is not None)

    if cached:
        data = json.loads(cached)
        data["cached"] = True
        print("Cache hit......returning answer")
        return data
    
    result = graph.invoke({
        "question": question,
        "context": "",
        "answer": "",
        "task": "",
        "sources":"",
        "history": conversation_history[-3:]
    })

    print("GRAPH result keys:", list(result.keys()))

    data = {"answer": result["answer"], "sources": result.get("sources", [])}
    
    
    try:
        cache.set(key, json.dumps(data), ex=CACHE_TTL)
    except redis.exceptions.RedisError as e:
        print(f"Error in caching {e}")

    save_convo(question=question, answer=data["answer"], task_type="qa")
    
    del conversation_history[:-3] #deletes the last 3 conversations saved to keep memory bounded


    conversation_history.append({
        "question": question,
        "answer": data["answer"]
    })
    data["cached"] = False
    return data

graph = build_graph()
if __name__ == "__main__":
    while True:
        question = input("\nAsk your knowledge base: ")
        if question.lower() == "quit":
            break
        print(f"\nAnswer: {ask(question)}")