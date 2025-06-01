## DEV
```bash
conda create -n rag-agent python=3.11
conda activate rag-agent

pip install langchain langgraph gradio langchain-community faiss-cpu langchain-openai pypdf

curl -H "Accept: application/json" "http://127.0.0.1:8080/search?q=agent&engines=google&format=json"

docker-compose up

# 設定回傳格式接受 html 與 json
sudo chown $USER:$USER ./searxng/settings.yml
sudo echo -e "\nsearch:\n  formats:\n    - html\n    - json" >> ./searxng/settings.yml

docker-compose down

OPENAI_BASE_URL = "http://10.1.1.68:11434"
OPENAI_API_KEY = "TEMPTY"
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-minilm:22m"

export OPENAI_BASE_URL=http://10.1.1.68:80/v1
export OPENAI_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDI3NDQ4NzAsIm5hbWUiOiJhZG1pbiJ9.dnn3Cl8LwJh7fFuLufARoz1evzEBKf9Gfr3n1hHDgN0
export CHAT_MODEL=llama3.2:3b
curl "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$CHAT_MODEL"'","messages": [{"role": "user","content": "Hello!"}]}'
```