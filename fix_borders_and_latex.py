#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт устранения серых полос и приведения LaTeX к строгому формату $$ ... $$
в Google Doc вкладке "Лаба 6000" (t.76ycwonhwvh).
1. Очистка всех paragraph borders (borderBottom, borderTop, borderBetween, borderLeft, borderRight, shading)
2. Очистка backgroundColor текста
3. Замена $$$$ на $$ через replaceAllText
"""

import json
import urllib.request

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

# 1. Загружаем структуру таба
url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
body = tab["documentTab"]["body"]["content"]
doc_end = body[-1]["endIndex"]

requests = []

# Шаг 1: Очистка backgroundColor по всему тексту таба
requests.append({
    "updateTextStyle": {
        "textStyle": {},
        "range": {
            "tabId": tab_id,
            "startIndex": 1,
            "endIndex": doc_end - 1
        },
        "fields": "backgroundColor"
    }
})

# Шаг 2: Очистка всех границ и заливок по каждому параграфу таба
for elem in body:
    if "paragraph" in elem:
        st = elem.get("startIndex")
        en = elem.get("endIndex")
        if st is not None and en is not None and st < en:
            requests.append({
                "updateParagraphStyle": {
                    "paragraphStyle": {},
                    "range": {
                        "tabId": tab_id,
                        "startIndex": st,
                        "endIndex": en
                    },
                    "fields": "borderBottom,borderTop,borderBetween,borderLeft,borderRight,shading"
                }
            })

# Шаг 3: Замена quadruple dollars $$$$ на double dollars $$
# В Google Docs API replaceAllText принимает tabsCriteria
requests.append({
    "replaceAllText": {
        "containsText": {
            "text": "$$$$",
            "matchCase": True
        },
        "replaceText": "$$",
        "tabsCriteria": {
            "tabIds": [tab_id]
        }
    }
})

print(f"Подготовлено {len(requests)} запросов. Отправка...")

batch_url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
req = urllib.request.Request(
    batch_url,
    data=json.dumps({"requests": requests}).encode("utf-8"),
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("Успешно выполнено! Ответ:", res.keys())
        for r in res.get("replies", []):
            if "replaceAllText" in r:
                print("Замен $$$$ на $$ выполнено:", r["replaceAllText"].get("occurrencesChanged", 0))
except urllib.error.HTTPError as e:
    print("Ошибка HTTP:", e.code, e.read().decode())
