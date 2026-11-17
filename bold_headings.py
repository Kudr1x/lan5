#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт выделения заголовков жирным начертанием (bold) во вкладке "Лаба 6000".
Все заголовки: Times New Roman, 14 pt, Bold, Black (#000000), выравнивание по левому краю (START).
"""

import json
import urllib.request
import re

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
body = tab["documentTab"]["body"]["content"]

heading_regex = re.compile(r'^\s*(\d+(\.\d+)*\.?\s+[А-ЯA-Z]|Цель работы|Теоретическая часть|Результаты|Заключение|Сводная таблица|Графический интерфейс)')

requests = []

for i, el in enumerate(body):
    p = el.get("paragraph")
    if not p:
        continue
    st = el.get("startIndex")
    en = el.get("endIndex")
    if st >= en:
        continue

    text = "".join(tr.get("textRun", {}).get("content", "") for tr in p.get("elements", [])).strip()
    named_style = p.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")

    is_heading = named_style.startswith("HEADING_") or bool(heading_regex.match(text))

    # Титульный лист: ОТЧЕТ и тема
    is_title_bold = (st < 800 and text in [
        "«САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ ЭЛЕКТРОТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ «ЛЭТИ» ИМ. В.И. УЛЬЯНОВА (ЛЕНИНА)»",
        "Отчет",
        "по лабораторной работе №5",
        "Тема: «Синхронизация хаотической системы Спротта B: классические методы Пикоры — Кэрола и быстрый обратимый метод»"
    ])

    if is_heading or is_title_bold:
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": True,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 14, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {
                    "tabId": tab_id,
                    "startIndex": st,
                    "endIndex": en
                },
                "fields": "bold,weightedFontFamily,fontSize,foregroundColor"
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": "CENTER" if is_title_bold else "START",
                    "lineSpacing": 150,
                    "indentFirstLine": {"magnitude": 0, "unit": "PT"},
                    "spaceAbove": {"magnitude": 12, "unit": "PT"},
                    "spaceBelow": {"magnitude": 6, "unit": "PT"}
                },
                "range": {
                    "tabId": tab_id,
                    "startIndex": st,
                    "endIndex": en
                },
                "fields": "alignment,lineSpacing,indentFirstLine,spaceAbove,spaceBelow"
            }
        })
        print(f"Heading: [{st}..{en}] {repr(text[:50])}")

print(f"Сформировано {len(requests)} запросов для заголовков.")

chunk_size = 50
for c_idx in range(0, len(requests), chunk_size):
    chunk = requests[c_idx:c_idx + chunk_size]
    req_body = {"requests": chunk}
    update_url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
    post_req = urllib.request.Request(
        update_url,
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(post_req) as resp:
        print(f"Порция {c_idx // chunk_size + 1} успешно применена: HTTP {resp.status}")

print("Все заголовки успешно оформлены по ГОСТу!")
