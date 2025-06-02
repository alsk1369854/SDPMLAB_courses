# RAG Agent
基於檢索增強生成(RAG)與大型語言模型(LLM)打造的AI代理，具網路搜尋與私有知識檢索功能，並且具有自我檢驗與逐步優化答案的能力。

## 架構
<image src="https://raw.githubusercontent.com/alsk1369854/SDPMLAB_courses/refs/heads/master/rag-agent/docs/workflow.png" alt="workflow.png">


## DEV
```bash
conda create -n rag-agent python=3.11
conda activate rag-agent

pip install langchain langgraph gradio langchain-community faiss-cpu langchain-openai pypdf
pip freeze > requirements.txt
```

## Deploy 
### Docker
```bash
# run
docker-compose up -d --build
docker-compose logs -f

# restart
docker-compose down
docker-compose up -d --build
```