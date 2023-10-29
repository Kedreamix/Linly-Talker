import requests

url = "http://ip:port"
headers = {
    "Content-Type": "application/json"
}
data = {
    "question": "北京有什么好玩的地方？"
}

response = requests.post(url, headers=headers, json=data)
# response_text = response.content.decode("utf-8")
answer, tag = response.json()
# print(answer)
if tag == 'success':
    response_text =  answer[0]
else:
    print("fail")
print(response_text)