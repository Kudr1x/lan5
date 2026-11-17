#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт точечной настройки стилей заголовков и обычного текста.
Убирает случайный bold со списков и пунктов описания, оставляя жирными ТОЛЬКО истинные заголовки.
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

# Истинные заголовки разделов и подразделов
true_heading_patterns = [
    r'^1\.\s+Цель работы',
    r'^2\.\s+Теоретическая часть',
    r'^2\.1\.\s+Исследуемая нелинейная',
    r'^2\.2\.\s+Симметричный обратимый',
    r'^2\.3\.\s+Методы непрерывной синхронизации',
    r'^2\.4\.\s+Теоретическое обоснование',
    r'^2\.5\.\s+Метод быстрой обратимой',
    r'^3\.\s+Результаты вычислительных',
    r'^3\.1\.\s+Исследование однонаправленной',
    r'^3\.2\.\s+Исследование двунаправленной',
    r'^3\.3\.\s+Исследование метода быстрой',
    r'^3\.4\.\s+Комплексные матрицы',
    r'^3\.5\.\s+Исследование параметров',
    r'^1\.\s+Зачем вообще нужно подбирать',
    r'^2\.\s+Как проводились вычисления',
    r'^3\.\s+Понятие машинного эпсилона',
    r'^1\.\s+Канал связи X:\s+точка',
    r'^2\.\s+Канал связи Y:\s+точка',
    r'^3\.\s+Канал связи Z:\s+точка',
    r'^4\.\s+Универсальная точка',
    r'^5\.\s+Сравнительный анализ бассейнов',
    r'^Сводные метрологические характеристики',
    r'^4\.\s+Сводная таблица результатов',
    r'^5\.\s+Графический интерфейс',
    r'^6\.\s+Заключение',
]

def is_true_heading(text):
    for pat in true_heading_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

title_patterns = [
    "МИНОБРНАУКИ РОССИИ",
    "Санкт-Петербургский государственный",
    "электротехнический университет",
    "«ЛЭТИ» им. В.И. Ульянова (Ленина)»",
    "Отчет",
    "по лабораторной работе №5",
    "по дисциплине «Моделирование нелинейных динамических систем»",
    "Тема: «Синхронизация хаотической системы Спротта B: классические методы Пикоры — Кэрола и быстрый обратимый метод»",
    "Санкт-Петербург",
    "2026"
]

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
    if not text:
        continue

    has_image = any("inlineObjectElement" in elem for elem in p.get("elements", []))

    if st <= 432:
        # Титульный лист (заканчивается годом "2026" на индексе 431)
        is_title_bold = text in [
            "«САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ ЭЛЕКТРОТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ «ЛЭТИ» ИМ. В.И. УЛЬЯНОВА (ЛЕНИНА)»",
            "Отчет",
            "по лабораторной работе №5",
            "Тема: «Синхронизация хаотической системы Спротта B: классические методы Пикоры — Кэрола и быстрый обратимый метод»"
        ]
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": is_title_bold,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 14, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "bold,weightedFontFamily,fontSize,foregroundColor"
            }
        })
    elif has_image:
        continue
    elif text.startswith("*Рис.") or text.startswith("Рис."):
        # Подпись к рисунку: курсив, по центру, без жирного
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": False,
                    "italic": True,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 12, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "bold,italic,weightedFontFamily,fontSize,foregroundColor"
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": "CENTER",
                    "lineSpacing": 150,
                    "indentFirstLine": {"magnitude": 0, "unit": "PT"},
                    "spaceAbove": {"magnitude": 2, "unit": "PT"},
                    "spaceBelow": {"magnitude": 8, "unit": "PT"}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "alignment,lineSpacing,indentFirstLine,spaceAbove,spaceBelow"
            }
        })
    elif (text.startswith("$$") or "\\begin{" in text or text.startswith("\\text{div}") or text.startswith("\\Phi")) and len(text) < 160:
        # Отдельная выключная формула (display math): по центру, без жирного
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": False,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 14, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "bold,weightedFontFamily,fontSize,foregroundColor"
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": "CENTER",
                    "lineSpacing": 150,
                    "indentFirstLine": {"magnitude": 0, "unit": "PT"}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "alignment,lineSpacing,indentFirstLine"
            }
        })
    elif is_true_heading(text):
        # Истинный заголовок: жирный, START
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": True,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 14, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "bold,weightedFontFamily,fontSize,foregroundColor"
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": "START",
                    "lineSpacing": 150,
                    "indentFirstLine": {"magnitude": 0, "unit": "PT"},
                    "spaceAbove": {"magnitude": 12, "unit": "PT"},
                    "spaceBelow": {"magnitude": 6, "unit": "PT"}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "alignment,lineSpacing,indentFirstLine,spaceAbove,spaceBelow"
            }
        })
    else:
        # Обычный текст параграфов: по ширине, отступ 1.25 см (36 пт), без bold
        requests.append({
            "updateTextStyle": {
                "textStyle": {
                    "bold": False,
                    "weightedFontFamily": {"fontFamily": "Times New Roman"},
                    "fontSize": {"magnitude": 14, "unit": "PT"},
                    "foregroundColor": {"color": {"rgbColor": {}}}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "bold,weightedFontFamily,fontSize,foregroundColor"
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": "JUSTIFIED",
                    "lineSpacing": 150,
                    "indentFirstLine": {"magnitude": 36, "unit": "PT"},
                    "spaceAbove": {"magnitude": 0, "unit": "PT"},
                    "spaceBelow": {"magnitude": 4, "unit": "PT"}
                },
                "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                "fields": "alignment,lineSpacing,indentFirstLine,spaceAbove,spaceBelow"
            }
        })

print(f"Всего запросов: {len(requests)}")

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

print("Все стили окончательно приведены в идеальное соответствие ГОСТу!")
