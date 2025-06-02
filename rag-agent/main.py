import random
def get_random_question() -> str:
    questions = [
        "今天高雄天氣如何？",
        "目前新台幣對美元匯率是多少？",
        "整理最近的 AI 相關新聞",
    ]
    return random.choice(questions)

print(get_random_question())