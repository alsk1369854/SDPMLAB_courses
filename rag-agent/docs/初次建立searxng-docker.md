# 設定回傳格式接受 html 與 json
```bash
sudo chown $USER:$USER ./searxng/settings.yml
sudo echo -e "\nsearch:\n  formats:\n    - html\n    - json" >> ./searxng/settings.yml

curl -H "Accept: application/json" "http://127.0.0.1:8080/search?q=agent&engines=google&format=json"
```