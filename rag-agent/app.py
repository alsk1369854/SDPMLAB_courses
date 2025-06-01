import gradio as gr
import uuid
from typing import Generator, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from modules.graph import create_graph, vector_store, text_spliter

graph = create_graph()
# for event in graph.stream({"question": "高雄天氣"}, stream_mode="debug"):
#     print(event)

def run_refresh_rag_index(files) -> str:
    # vector_store.index.reset() # 清空向量索引
    # vector_store.docstore._dict.clear()  # 清空文檔存儲
    # vector_store.index_to_docstore_id.clear()  # 清空索引對應表
    for file_path in files:
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
            vector_store.add_documents(documents=chunks, ids=uuids)
        except Exception as e:
            return f"Error: {e}"

    # vector_store.save_local("faiss_db")
    return "Sucessfully"

def run_agent(question: str, history: list[list[str]]) -> Generator[tuple[str, str], Any, None]:
    inputs = {"question": question, "max_documents": 5}

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
                context = ""
                for i, doc in enumerate(documents):
                    source_type = 'web' if 'link' in doc.get('metadata') else 'retrieval'
                    text = doc["page_content"]
                    context += f"{i+1}. Document(type={source_type}, text={text})\n"
                debug_trace += f"\n\n@Documents:\n{context}\n"

        elif event_type == "task_result":
            state = {k:v for k, v in payload["result"]}

            # 追蹤 Agent 的回答生成
            if node_name == "answer_generation":
                answer = state["answer"]
                debug_trace += f"\n\n@{node_title}\n"
                debug_trace += f"Answer:\n{answer}\n"
                final_answer = answer

            # 追蹤 Agent 的回答檢查
            if node_name == "answer_guard":
                retry = state["retry"]
                explanation = state["explanation"]
                debug_trace += f"\n\n@{node_title}\n"
                debug_trace += f"Retry: {retry}\n"
                debug_trace += f"Explanation:\n{explanation}\n"
                final_answer = answer

        yield final_answer, debug_trace

    debug_trace += "\n\n===== Trace End =====\n\n"
    yield final_answer, debug_trace

# Gradio Interface setup
with gr.Blocks() as demo:
    gr.Markdown("# RAG Agent Interface")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Agent Chat")
            trace_chatbot = gr.Chatbot(label="Chat", type="messages")
            question_input = gr.Textbox(show_label=False, placeholder="Enter your question...")

            def on_input_submit(question: str, history: list[dict]):
                return "", [{"role": "user", "content": question}]

            def wrapped_run_agent(history: list[dict]) -> Generator[dict, Any, None]:
                question = history[0]["content"]
                history.append({"role": "assistant", "content": ""})
                final_ans = ""
                for final_ans, debug_trace in run_agent(question, history):
                    history[-1]['content'] = debug_trace
                    final_ans = final_ans
                    yield history

                history[-1]['content'] += f"\n@Final Answer:\n{final_ans}\n"
                yield history

            question_input.submit(
                on_input_submit,
                inputs=[question_input, trace_chatbot],
                outputs=[question_input, trace_chatbot],
                queue=False
            ).then(
                wrapped_run_agent,
                inputs=trace_chatbot,
                outputs=trace_chatbot,
            )

        with gr.Column(scale=1):
            gr.Markdown("## Knowledge Files")
            file_upload = gr.File(label="Upload Knowledge Documents (.pdf, .txt, .md)", file_count="multiple", type="filepath")
            refresh_status = gr.Textbox(label="Refresh Status", interactive=False)
            refresh_button = gr.Button("Refresh RAG Index")
            refresh_button.click(fn=run_refresh_rag_index, inputs=file_upload, outputs=refresh_status)

demo.launch(server_name="0.0.0.0", server_port=18080, share=True)