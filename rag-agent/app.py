import gradio as gr
import uuid
import faiss
import os
import random
from typing import Generator, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from modules.graph import create_graph, text_spliter, embed

# ROOT_PATH 資源請求的根路徑: 適用於向代理多服務情境
ROOT_PATH = os.environ.get("ROOT_PATH", None)

graph = create_graph()

def update_rag_docs(vector_store, file_paths: list[str]) -> None:
    for file_path in file_paths:
        file_name = file_path.split("/")[-1]
        try:
            if file_path.lower().endswith(".txt") or file_path.lower().endswith(".md"):
                loader = TextLoader(file_path)
            elif file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue  # Skip unsupported file types

            documents = loader.load()
            chunks = text_spliter.split_documents(documents)
            uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            vector_store.add_documents(documents=chunks, ids=uuids, matedate={"file_name": file_name})
        except Exception as e:
            return f"Error: {e}"

def create_vector_store(file_paths: list[str]) -> FAISS:
    vector_store = FAISS(
        embedding_function=embed,
        index=faiss.IndexFlatL2(len(embed.embed_query("index"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    update_rag_docs(vector_store, file_paths)
    return vector_store

def run_agent(vector_store: FAISS, question: str) -> Generator[tuple[str, str], Any, None]:
    inputs = {
        "question": question, 
        "vector_store": vector_store,
        "max_reference_documents": 5, 
    }

    docs_tracked = False
    debug_trace = "\n\n===== Trace Start =====\n\n"
    final_answer = ""
    for event in graph.stream(inputs, stream_mode="debug"):
        event_type = event["type"]
        payload = event["payload"]
        node_name = payload["name"]
        node_title = node_name.replace("_", " ").title()

        if event_type == "task":
            state = payload["input"].model_dump()

            # 追蹤 Agent 的參考文件
            if node_name == "answer_generation" and not docs_tracked:
                docs_tracked = True
                documents = state["documents"]

                doc_texts = []
                for i, doc in enumerate(documents):
                    source_from = "web" if "link" in doc.get("metadata") else "local"
                    doc_texts.append(f"{i+1}. Document(from={source_from}, text={doc.get('page_content', '')})")

                context = '\n\n'.join(doc_texts)
                debug_trace += f"\n\n@ Documents:\n{context}"


        elif event_type == "task_result":
            state = {k:v for k, v in payload["result"]}

            if node_name == "retrieve_documents":
                query = state["retrieve_query"]
                debug_trace += f"\n\n@ {node_title}\n"
                debug_trace += f"Query:\n{query}\n"

            if node_name == "web_search":
                query = state["web_search_query"]
                debug_trace += f"\n\n@ {node_title}\n"
                debug_trace += f"Query:\n{query}\n"

            # 追蹤 Agent 的回答生成
            if node_name == "answer_generation":
                answer = state["answer"]
                debug_trace += f"\n\n@ {node_title}\n"
                debug_trace += f"Answer:\n{answer}\n"
                final_answer = answer

            # 追蹤 Agent 的回答檢查
            if node_name == "answer_guard":
                retry = state["retry"]
                explanation = state["explanation"]
                debug_trace += f"\n\n@ {node_title}\n"
                debug_trace += f"Retry: {retry}\n"
                debug_trace += f"Explanation:\n{explanation}\n"
                final_answer = answer

        yield final_answer, debug_trace

    debug_trace += "\n\n===== Trace End =====\n\n"
    yield final_answer, debug_trace

def get_random_question() -> str:
    questions = [
        "今天高雄天氣如何？",
        "目前新台幣對美元匯率是多少？",
    ]
    return random.choice(questions)

# Gradio Interface setup
with gr.Blocks(title="RAG Agent Demo") as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(f"# RAG Agent Interface\nRAG Agent 是一款結合大型語言模型(LLM)與檢索增強生成(RAG)的智慧代理，具備網路搜尋、知識檢索與答案優化能力。")
        with gr.Column(scale=1):
            gr.HTML("""
                <a href="https://github.com/alsk1369854/SDPMLAB_courses/tree/master/rag-agent" target="_blank" style="text-decoration: none;">
                    <button style="
                        background-color: #24292e;
                        color: white;
                        width: 100%;
                        padding: 10px 20px;
                        font-size: 16px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    ">
                        🔗 View on GitHub
                    </button>
                </a>
            """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Agent Chat")
            final_answer = gr.Textbox(show_label=True, label="Final Answer", interactive=False)
            trace_chatbot = gr.Chatbot(label="Agent Trace", type="messages")
            question_input = gr.Textbox(show_label=True, label="Question", placeholder="Enter your question...", value=get_random_question())

        with gr.Column(scale=1):
            gr.Markdown("## Knowledge Files")
            use_demo_btn = gr.Button("Use Demo File")
            file_upload = gr.File(label="Upload Knowledge Documents (.pdf, .txt, .md)", file_count="multiple", type="filepath")
    
    def on_download_demo_click() -> tuple[str, list[str]]:
        return  "有哪些課程是由資訊工程學系開設的？", ["./學年開課資訊_RAG_DEMO.md"]
    use_demo_btn.click(on_download_demo_click, None, [question_input, file_upload])

    def on_question_input_submit(question: str, history: list[dict]) -> tuple[str, list[dict]]:
        return get_random_question(), [{"role": "user", "content": question}]
    
    def answer_generator(file_paths: list[str] | None, history: list[dict]) -> Generator[tuple[str, list[dict]], None, None]:
        file_paths =  [] if file_paths is None else file_paths
        vector_store = create_vector_store(file_paths)
        question = history[0]["content"]
        history.append({"role": "assistant", "content": ""})
        for final_ans, debug_trace in run_agent(vector_store, question):
            history[-1]['content'] = debug_trace
            final_ans = final_ans
            yield "", history

        history[-1]['content'] += f"\n@ Final Answer:\n{final_ans}\n"
        yield final_ans, history

    question_input.submit(
        on_question_input_submit,
        inputs=[question_input, trace_chatbot],
        outputs=[question_input, trace_chatbot],
        queue=False,
    ).then(
        answer_generator,
        inputs=[file_upload, trace_chatbot],
        outputs=[final_answer, trace_chatbot],
    )

if __name__ == "__main__":
    # for event in graph.stream({"question": "今天高雄天氣如何？"}, stream_mode="debug"):
    #     print(event)

    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=18080, 
        share=True,
        root_path=ROOT_PATH
    )