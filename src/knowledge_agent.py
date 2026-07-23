from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import chromadb
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv
from src.database import save_convo, get_recent, init_db
load_dotenv()
import redis
import json
import os

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")


redis_url = os.getenv("REDIS_URL")
llm = ChatGroq(model="llama-3.1-8b-instant",api_key= os.getenv("GROQ_API_KEY"), temperature=0)
cache = redis.from_url(redis_url,decode_responses=True)
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    task: str
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
    results = collection.query(
        query_texts=[state["question"]],
        n_results=3
    )
    context = "\n".join(results["documents"][0])
    return {"context": context}

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

def ask(question: str) -> str:
    graph = build_graph()
    key = question.lower().strip()

    cached = cache.get(key)

    if cached:
        print("Cache hit......returning answer")
        return cached
    result = graph.invoke({
        "question": question,
        "context": "",
        "answer": "",
        "task": "",
        "history": conversation_history
    })
     
    answer = result["answer"]
    cache.set(key, answer, ex=3600)

    save_convo(question=question, answer=answer, task_type="qa")


    conversation_history.append({
        "question": question,
        "answer": result["answer"]
    })

    return result["answer"]

if __name__ == "__main__":
    while True:
        question = input("\nAsk your knowledge base: ")
        if question.lower() == "quit":
            break
        print(f"\nAnswer: {ask(question)}")