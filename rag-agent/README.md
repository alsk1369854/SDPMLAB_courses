## DEV
```bash
conda create -n rag-agent python=3.11
conda activate rag-agent

pip install langchain langgraph gradio langchain-community faiss-cpu langchain-openai pypdf
pip freeze > requirements.txt
```

## Docker
### Run
```bash
docker-compose up -d --build
docker-compose logs -f
```

### Restart
```bash
docker-compose down
docker-compose up -d --build
```