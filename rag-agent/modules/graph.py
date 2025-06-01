import os
import faiss
from typing import List, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.utilities import SearxSearchWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from modules.embeddings import CustomOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
CHAT_MODEL =  os.environ.get("CHAT_MODEL", "chatgpt-3.5-turbo")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

SEARXNG_BASE_URL = "http://127.0.0.1:8080"
OPENAI_BASE_URL = "http://10.1.1.68:80/v1"
OPENAI_API_KEY = ""
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-minilm:22m"

# 大語言模型
llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
# 嵌入模型
embed = CustomOpenAIEmbeddings(model=EMBED_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# 文本分塊器
text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, # 每 1000 tokens 為一個 chunk
    chunk_overlap=200, # 每個 chunk 之間重疊 200 tokens
)
# 向量資料庫
vector_store = FAISS(
    embedding_function=embed,
    index=faiss.IndexFlatL2(len(embed.embed_query("index"))),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

class GraphState(BaseModel):
    question: str # User question

    documents: list[Document] = Field(default_factory=list) # List of retrieved documents
    max_documents: int = 5 # Maximum number of documents to retrieve

    answer: str = "" # Final answer
    explanation: str = "" # Useful explanation
    retry: bool = False # Answer Regenerate Control
    max_retries: int = 5 # Maximum number of answer to generated


def get_environment_message() -> str:
    return ""


# 網路搜尋工具
searxng_search = SearxSearchWrapper(searx_host=SEARXNG_BASE_URL)
@tool
def search_tool(query: str, k: int = 5) -> list[Document]:
    """web search tool"""
    results = searxng_search.results(
        query,
        num_results=k,
        language="zh-TW",
        engines=["google", "wiki", "arxiv"],
    )
    documents = [
        Document(
            page_content=result["snippet"],
            metadata={
                "title": result["title"],
                "link": result["link"],
                "engines": result["engines"],
                "category": result["category"]
            }
        )
        for result in results
    ]
    return documents


# RAG 檢索工具
@tool
def retrieve_tool(query: str, k: int = 5) -> list[Document]:
    """documents retrieve tool"""
    return vector_store.similarity_search(query, k=k)

# 知識文件檢索節點
def retrieve_documents(state: GraphState):
    question = state.question
    max_documents = state.max_documents

    retrieved_documents = retrieve_tool.invoke({"query": question, "k": max_documents})
    return {"documents": retrieved_documents}


# 知識文件守衛節點
RELATED_GUARD_INSTRUCTION = """\
你是個專業的評估員，負責評估檢索到的文件內容與使用者問題的相關性。
如果文件包含與問題相關的關鍵字或語義，則將其評為相關。

請仔細、客觀地評估文件內容是否至少包含一些與問題相關的資訊。
"""

RELATED_GUARD_PROMPT_TEMPLATE = """\
這是檢索到的文件內容:
{document}

這是使用者問題:
{question}
"""

RELATED_GUARD_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", RELATED_GUARD_INSTRUCTION),
        ("human", RELATED_GUARD_PROMPT_TEMPLATE),
])


class RelatedGuardOutput(BaseModel):
    related: Literal["yes", "no"] = Field(
        description="文件內容是否包含至少一些能夠回答問題的資訊"
    )

def related_guard(state: GraphState):
    question = state.question
    documents = state.documents

    llm_with_struct = llm.with_structured_output(RelatedGuardOutput)
    filtered_docs = []
    for doc in documents:
        messages = RELATED_GUARD_CHAT_TEMPLATE.invoke({
            "document": doc,
            "question": question,
        }).to_messages()
        result = llm_with_struct.invoke(messages)
        if result.related.lower() == "yes":
            filtered_docs.append(doc)

    return {"documents": filtered_docs}

# 網路查詢節點
def web_search(state: GraphState):
    question = state.question
    documents = state.documents
    max_documents = state.max_documents

    documents_len = len(documents)
    if documents_len >= max_documents:
        return {}

    search_docs = search_tool.invoke({"query": question, "k": max_documents - documents_len})
    documents.extend(search_docs)
    return {"documents": documents}

# 回答生成節點
ANSWER_GENERATION_INSTRUCTION = """\
你是一個優秀的問答助手，任務是仔細閱讀參考資料內容後回答使用者的問題。
如果參考資料不足以回答這個問題，請直接回答“查無相關資料”。
確保最終答案的簡潔，直接回覆最終回答。
"""

ANSWER_GENERATION_PROMPT_TEMPLATE = """\
# 參考資料
{context}

# 使用者問題:
{question}

你的回答是:
"""

ANSWER_GENERATION_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", ANSWER_GENERATION_INSTRUCTION),
        ("human", ANSWER_GENERATION_PROMPT_TEMPLATE),
])

def answer_generation(state: GraphState):
    question = state.question
    documents = state.documents
    max_retries = state.max_retries

    context = "\n".join([f"Document({doc.page_content})" for doc in documents])
    messages = ANSWER_GENERATION_CHAT_TEMPLATE.invoke({
        "context": context,
        "question": question,
    }).to_messages()
    answer = llm.invoke(messages).content
    return {
        "max_retries": max_retries - 1,
        "answer": answer,
    }

# 參考知識文件檢查節點
def documents_router(state: GraphState):
    documents = state.documents
    max_documents = state.max_documents
    if len(documents) < max_documents:
        return "not_enough"
    else:
        return "enough"
    
# 答覆檢查節點
ANSWER_GUARD_INSTRUCTION = """\
你是一位正在批改測驗的老師，你將得到一個問題和一個學生的答案。

以下是需要遵循的評判標準：
1. 學生的回答有正確回答了問題

評判結果：
- 結果為 "yes" 表示學生的答案符合所有標準。這是您能給的最高分數。如果學生的答案在滿足問題的情況下額外又給出了其他說明，則他能夠獲得 "yes"。
- 結果為 "no" 表示學生的答案不符合所有標準。這是您能給的最低分數。

逐步解釋你的推理，以確保你的推理和結論是正確的。
避免一開始就簡單陳述正確答案。
"""

ANSWER_GUARD_PROMPT_TEMPLATE = """\
問題:
{question}

學生回答:
{answer}
"""

ANSWER_GUARD_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", ANSWER_GUARD_INSTRUCTION),
        ("human", ANSWER_GUARD_PROMPT_TEMPLATE),
])

class AnswerGuardOutput(BaseModel):
    useful: Literal["yes", "no"] = Field(
        description="對學生答案的評判結果"
    )
    explanation: str = Field(
        description="對評判結果的解釋"
    )

def answer_guard(state: GraphState):
    question = state.question
    answer = state.answer

    llm_with_struct = llm.with_structured_output(AnswerGuardOutput)
    messages = ANSWER_GUARD_CHAT_TEMPLATE.invoke({
        "answer": answer,
        "question": question,
    }).to_messages()
    result = llm_with_struct.invoke(messages)
    explanation = result.explanation
    if result.useful.lower() == "yes":
        return {"retry": False, "explanation": explanation}
    else:
        return {"retry": True, "explanation": explanation}

def answer_guard_router(state: GraphState):
    retry = state.retry
    max_retries = state.max_retries
    if max_retries <= 0:
        return END
    if retry:
        return "regenerate"
    else:
        return END
    
# 建立工作流
def create_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("related_guard", related_guard)
    workflow.add_node("web_search", web_search)
    workflow.add_node("answer_generation", answer_generation)
    workflow.add_node("answer_guard", answer_guard)

    # Build graph
    workflow.add_edge(
        START,
        "retrieve_documents"
    )
    workflow.add_edge(
        "retrieve_documents",
        "related_guard"
    )
    workflow.add_conditional_edges(
        "related_guard",
        documents_router,
        {
            "enough": "answer_generation",
            "not_enough": "web_search",
        },
    )
    workflow.add_edge(
        "web_search",
        "answer_generation"
    )
    workflow.add_edge(
        "answer_generation",
        "answer_guard"
    )
    workflow.add_conditional_edges(
        "answer_guard",
        answer_guard_router,
        {
            "regenerate": "answer_generation",
            END: END,
        },
    )

    # Compile
    graph = workflow.compile()
    return graph