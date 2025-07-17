import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SearxSearchWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from modules.embeddings import CustomOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
CHAT_MODEL =  os.environ.get("CHAT_MODEL", "chatgpt-3.5-turbo")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# SEARXNG_BASE_URL = "http://127.0.0.1:8080"
# OPENAI_BASE_URL = "http://10.1.1.68:80/v1"
# OPENAI_API_KEY = "..."
# CHAT_MODEL = "gemma3:12b"
# EMBED_MODEL = "all-minilm:22m"

# 大語言模型
llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
# 嵌入模型
embed = CustomOpenAIEmbeddings(model=EMBED_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# 文本分塊器
text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, # 每 1000 tokens 為一個 chunk
    chunk_overlap=200, # 每個 chunk 之間重疊 200 tokens
)
# # 向量資料庫
# vector_store = FAISS(
#     embedding_function=embed,
#     index=faiss.IndexFlatL2(len(embed.embed_query("index"))),
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

class GraphState(BaseModel):
    question: str # User question
    vector_store: Any

    retrieve_query: str = "" # Query for retrieving documents
    web_search_query: str = "" # Query for web search
    documents: list[Document] = Field(default_factory=list) # List of retrieved documents
    max_reference_documents: int = 5 # Maximum number of documents to retrieve

    answer: str = "" # Final answer
    explanation: str = "" # Useful explanation
    retry: bool = False # Answer Regenerate Control
    max_retries: int = 5 # Maximum number of answer to generated


ENRVIRONMENT_TEMPLATE = """\
當前環境資訊:

# 對話語系:
台灣

# 當前時間:
{time_info}
"""
def get_environment() -> str:
    # 取得當前時間與時區
    tz = ZoneInfo("Asia/Taipei")
    now = datetime.now(tz)

    # formatted_time = now.strftime("%-m/%-d/%Y, %-I:%M:%S %p")
    # offset = now.strftime('%z')  # e.g., +0800
    # formatted_offset = f"{offset[:3]}:{offset[3:]}"
    # tz_info = f"({now.tzinfo}, UTC{formatted_offset})"
    # return f"#當前環境資訊：\n\n## 語系:\n台灣\n\n## 當前時間:\n{formatted_time} {tz_info}\n"

    formatted_time = now.strftime("%Y/%m/%d, %H:%M:%S, UTC %z")
    return ENRVIRONMENT_TEMPLATE.format(time_info=formatted_time)


# 網路搜尋工具
searxng_search = SearxSearchWrapper(searx_host=SEARXNG_BASE_URL)
@tool
def search_tool(query: str, k: int=5) -> list[Document]:
    """web search tool"""
    try:
        results = searxng_search.results(
            query,
            num_results=k,
            language="zh-TW",
            engines=["google", "wiki"],
        )
    except Exception as e:
        results = []
        print(f"Error web search: {e}")
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
def retrieve_tool(vector_store: FAISS ,query: str, k: int = 5) -> list[Document]:
    """documents retrieve tool"""
    docs = vector_store.similarity_search(query, k=k)
    text_set = set()
    filtered_docs = []
    for doc in docs:
        if doc.page_content in text_set:
            continue
        text_set.add(doc.page_content)
        filtered_docs.append(doc)
    return filtered_docs[:k]


# 知識文件檢索節點
RETRIEVE_QUERY_INSTRUCTION = """\
你是一位專業的知識檢索助手，負責協助將使用者的自然語言問題轉換為一組最適合文件查找的檢索關鍵詞。

請遵循以下原則來擬定檢索關鍵詞：
1. 僅保留具有語義意涵的「主題名詞」、「重要技術詞彙」、「關鍵概念」。
2. 移除無意義的助詞、疑問詞、冗長動詞（如「如何實作」、「要不要」、「什麼是」）。
3. 以「繁體中文關鍵詞」為主，必要時可補充英文術語。
4. 多個關鍵詞請以英文逗號（,）分隔。

# 範例
問題：如何設計具補償機制的分散式交易流程，並結合 Saga 模式？\
檢索關鍵詞：分散式交易, 補償機制, Saga 模式, 微服務, 交易一致性

問題：用 Python 實作 YOLOv8 做小物件瑕疵偵測的最佳做法是什麼？\
檢索關鍵詞：Python, YOLOv8, 小物件偵測, 瑕疵檢測, 實作方法

問題：什麼是 BERT 模型的 attention 機制？\
檢索關鍵詞：BERT, attention 機制, 自注意力
"""

RETRIEVE_QUERY_PROMPT_TEMPLATE = """\
使用者問題：
{question}
"""

RETRIEVE_QUERY_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", RETRIEVE_QUERY_INSTRUCTION),
        ("human", RETRIEVE_QUERY_PROMPT_TEMPLATE),
        ("human", "{environment}"),
])

class RetrieveQueryOutput(BaseModel):
    query: str = Field(
        description="檢索關鍵詞"
    )

def retrieve_documents(state: GraphState):
    question = state.question
    vector_store = state.vector_store
    max_reference_documents = state.max_reference_documents

    # 生成檢索關鍵詞
    llm_with_struct = llm.with_structured_output(RetrieveQueryOutput)
    environment = get_environment()
    messages = RETRIEVE_QUERY_CHAT_TEMPLATE.invoke({
        "question": question,
        "environment": environment,
    }).to_messages()
    answer = llm_with_struct.invoke(messages)
    
    # 根據關鍵詞檢所本地知識庫
    retrieved_documents = retrieve_tool.invoke({"vector_store": vector_store, "query": answer.query, "k": max_reference_documents})
    return {"documents": retrieved_documents, "retrieve_query": answer.query}


# 知識文件守衛節點
RELATED_GUARD_INSTRUCTION = """\
你是一位專業的知識評估員，負責判斷一段文件內容是否對回答問題有所幫助。

評估準則：
1. 是否有助於用來回答問題（即使不是完整答案）？

若符合上述任何條件，請視為「相關」。
"""

RELATED_GUARD_PROMPT_TEMPLATE = """\
# 文件內容：
{document}

# 問題：
{question}
"""

RELATED_GUARD_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", RELATED_GUARD_INSTRUCTION),
        ("human", RELATED_GUARD_PROMPT_TEMPLATE),
        ("human", "{environment}"),
])

class RelatedGuardOutput(BaseModel):
    related: Literal["yes", "no"] = Field(
        description="是否相關"
    )

def related_guard(state: GraphState):
    question = state.question
    documents = state.documents

    llm_with_struct = llm.with_structured_output(RelatedGuardOutput)
    environment = get_environment()
    filtered_docs = []
    for doc in documents:
        
        # 檢查檢索結果是否與問題相關
        messages = RELATED_GUARD_CHAT_TEMPLATE.invoke({
            "document": doc,
            "question": question,
            "environment": environment,
        }).to_messages()
        anser = llm_with_struct.invoke(messages)
        
        # 加入至候選參考知識文件中
        if anser.related.lower() == "yes":
            filtered_docs.append(doc)

    return {"documents": filtered_docs}


# 網路查詢節點
WEB_SEARCH_QUERY_INSTRUCTION = """\
你是一位專業的網路搜尋助手，負責根據問題，擬定一個適合送出到搜尋引擎（如 Google）的中文查詢語句。

請遵守以下原則：
1. 查詢語句應簡潔明確，保留核心關鍵詞與語意重點。
2. 排除冗詞（如「請問」、「幫我查一下」等），使用搜尋引擎慣用的簡明語法。
3. 若有具體主題、技術、年份、地點，可酌情加入以提升準確度。

# 範例
問題：什麼是 LoRA 微調？\
查詢語句：LoRA 微調 是什麼

問題：2024 年有哪些生成式 AI 應用趨勢？\
查詢語句：2024 生成式 AI 趨勢

問題：如何用 Python 抓取網頁資料？\
查詢語句：Python 網頁爬蟲 教學
"""

WEB_SEARCH_QUERY_PROMPT_TEMPLATE = """\
問題:
{question}
"""

WEB_SEARCH_QUERY_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", WEB_SEARCH_QUERY_INSTRUCTION),
        ("human", WEB_SEARCH_QUERY_PROMPT_TEMPLATE),
        ("human", "{environment}"),
])

class WebSearchQueryOutput(BaseModel):
    query: str = Field(
        description="查詢語句"
    )

def web_search(state: GraphState):
    question = state.question
    documents = state.documents
    max_reference_documents = state.max_reference_documents

    # 檢查先前檢所的私有文件是否已滿足參考文件數量
    documents_len = len(documents)
    if documents_len >= max_reference_documents:
        return {}
    
    # 生成網路搜尋查詢語句
    llm_with_struct = llm.with_structured_output(WebSearchQueryOutput)
    environment = get_environment()
    messages = WEB_SEARCH_QUERY_CHAT_TEMPLATE.invoke({
        "question": question,
        "environment": environment,
    }).to_messages()
    answer = llm_with_struct.invoke(messages)
    
    # 根據查詢語句進行網路搜尋
    search_docs = search_tool.invoke({"query": answer.query, "k": max_reference_documents - documents_len})
    documents.extend(search_docs)
    return {"documents": documents, "web_search_query": answer.query}

# 回答生成節點
ANSWER_GENERATION_INSTRUCTION = """\
你是一位專業的問答助手，負責根據提供的「參考資料」來回答問題。

請遵守以下原則：
1. 僅根據參考資料內容作答，禁止編造未出現的資訊。
2. 若資料不足，請明確回覆「查無相關資料」。
3. 回答要簡潔、明確，避免冗言贅語。
"""

ANSWER_GENERATION_PROMPT_TEMPLATE = """\
# 前次回答（若有）：
{prev_answer}

# 對於前次回答的修改建議（若有）：
{prev_answer_suggestion}

# 參考資料：
{context}

# 問題：
{question}

# 回答：
"""

ANSWER_GENERATION_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", ANSWER_GENERATION_INSTRUCTION),
        ("human", ANSWER_GENERATION_PROMPT_TEMPLATE),
        ("human", "{environment}"),
])

def answer_generation(state: GraphState):
    question = state.question
    documents = state.documents
    prev_answer = state.answer
    prev_answer_suggestion = state.explanation
    max_retries = state.max_retries

    # 將檢索結果整理為參考資料格式
    doc_texts = []
    for doc in documents:
        doc_from = "web" if "link" in doc.metadata else "local"
        doc_texts.append(f"Document(from={doc_from}, text={doc.page_content})")
    context = "\n".join(doc_texts)
    
    # 生成回答
    environment = get_environment()
    messages = ANSWER_GENERATION_CHAT_TEMPLATE.invoke({
        "prev_answer": prev_answer,
        "prev_answer_suggestion": prev_answer_suggestion,
        "context": context,
        "question": question,
        "environment": environment,
    }).to_messages()
    answer = llm.invoke(messages).content
    return {
        "max_retries": max_retries - 1,
        "answer": answer,
    }

# 參考知識文件檢查節點
def documents_router(state: GraphState):
    documents = state.documents
    max_reference_documents = state.max_reference_documents
    if len(documents) < max_reference_documents:
        return "not_enough"
    else:
        return "enough"
    
# 答覆檢查節點
ANSWER_GUARD_INSTRUCTION = """\
你是一位老師，負責根據「問題」和「學生的參考資料」來評斷學生回答是否符合標準。

請依據以下標準進行評分：
1. 回答是否回應了問題的核心？
2. 回答是否符合參考資料內容？
3. 是否存在明顯的錯誤、誤解或答非所問的情形？

評分選項：
- "yes"：表示學生的答案符合所有標準。
- "no"：表示學生的答案不符合所有標準。請在解釋中附上具體的修改建議，協助學生修正答案。
"""

ANSWER_GUARD_PROMPT_TEMPLATE = """\
# 問題：
{question}

# 學生的參考資料：
{context}

# 學生回答：
{answer}
"""

ANSWER_GUARD_CHAT_TEMPLATE = ChatPromptTemplate([
        ("system", ANSWER_GUARD_INSTRUCTION),
        ("human", ANSWER_GUARD_PROMPT_TEMPLATE),
        ("human", "{environment}"),
])

class AnswerGuardOutput(BaseModel):
    result: Literal["yes", "no"] = Field(
        description="評分結果"
    )
    explanation: str = Field(
        description="評分說明"
    )

def answer_guard(state: GraphState):
    question = state.question
    answer = state.answer
    documents = state.documents

    # 將檢索結果整理為參考資料格式
    doc_texts = []
    for doc in documents:
        doc_from = "web" if "link" in doc.metadata else "local"
        doc_texts.append(f"Document(from={doc_from}, text={doc.page_content})")
    context = "\n".join(doc_texts)

    # 檢查本次生成的回答是否有效
    environment = get_environment()
    messages = ANSWER_GUARD_CHAT_TEMPLATE.invoke({
        "question": question,
        "context": context,
        "answer": answer,
        "environment": environment,
    }).to_messages()
    llm_with_struct = llm.with_structured_output(AnswerGuardOutput)
    result = llm_with_struct.invoke(messages)
    
    # 根據評分結果決定是否需要重新生成回答
    explanation = result.explanation
    if result.result.lower() == "yes":
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
    # 條件邊：檢索結果是否足夠
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
    # 條件邊：回答是否滿足要求
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