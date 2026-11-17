#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт полного ГОСТ-форматирования вкладки "Лаба 6000" (t.76ycwonhwvh).
Требования:
- Шрифт: Times New Roman
- Размер: 14 пт (для основного текста)
- Межстрочный интервал: 1.5 строки (150%)
- Цвет текста: ТОЛЬКО ЧЕРНЫЙ (#000000), никакой серости!
- Выравнивание основного текста: по ширине (JUSTIFIED)
- Красная строка: отступ первой строки 1.25 см (36 пт)
- Заголовки: по левому краю (START), жирные, черный цвет
- Формулы LaTeX ($$ ... $$): по центру (CENTER), на отдельной строке
- Рисунки и подписи к рисункам: по центру (CENTER)
"""

import json
import urllib.request
import re

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

# 1. Получаем структуру документа
url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
body = tab["documentTab"]["body"]["content"]
doc_end = body[-1]["endIndex"]

print(f"Таб 6000: всего элементов {len(body)}, диапазон от 1 до {doc_end}")

requests = []

# ---------------------------------------------------------------------------
# ЭТАП 1: ГЛОБАЛЬНЫЙ СТИЛЬ ТЕКСТА (Times New Roman, 14 pt, Черный #000000)
# ---------------------------------------------------------------------------
requests.append({
    "updateTextStyle": {
        "textStyle": {
            "weightedFontFamily": {"fontFamily": "Times New Roman"},
            "fontSize": {"magnitude": 14, "unit": "PT"},
            "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}}
        },
        "range": {
            "tabId": tab_id,
            "startIndex": 1,
            "endIndex": doc_end - 1
        },
        "fields": "weightedFontFamily,fontSize,foregroundColor"
    }
})

# ---------------------------------------------------------------------------
# ЭТАП 2: ПОПАРАГРАФНОЕ ФОРМАТИРОВАНИЕ (Выравнивание, интервал 1.5, отступы)
# ---------------------------------------------------------------------------

title_center_patterns = [
    "МИНОБРНАУКИ РОССИИ",
    "Санкт-Петербургский государственный",
    "электротехнический университет",
    "«ЛЭТИ» им. В.И. Ульянова",
    "Кафедра САПР",
    "Кафедра систем",
    "Отчет",
    "по лабораторной работе №5",
    "по дисциплине «Моделирование нелинейных динамических систем»",
    "Тема: «Синхронизация",
    "Санкт-Петербург",
    "2026"
]

heading_regex = re.compile(r'^\s*(\d+(\.\d+)*\.?\s+[А-ЯA-Z]|Цель работы|Теоретическая часть|Результаты|Заключение|Сводная таблица|Графический интерфейс|Выполнил|Проверил)')

for i, el in enumerate(body):
    p = el.get("paragraph")
    if not p:
        continue

    st = el.get("startIndex")
    en = el.get("endIndex")
    if st >= en:
        continue

    named_style = p.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
    text = "".join(tr.get("textRun", {}).get("content", "") for tr in p.get("elements", [])).strip()
    has_image = any("inlineObjectElement" in elem for elem in p.get("elements", []))

    # Определяем тип параграфа:
    is_title_page = (st < 800)
    is_heading = named_style.startswith("HEADING_") or bool(heading_regex.match(text))
    is_formula = ("$$" in text) or ("\\begin{" in text) or ("\\dfrac{" in text) or ("\\text{div}" in text) or ("\\Phi" in text)
    is_caption = text.startswith("Рис.") or text.startswith("*Рис.") or text.startswith("Таблица ")
    is_bullet = text.startswith("•") or text.startswith("-") or text.startswith("1.") or text.startswith("2.") or text.startswith("3.") or text.startswith("4.") or text.startswith("5.")

    p_style = {
        "lineSpacing": 150  # 1.5 межстрочный интервал по ГОСТ
    }
    fields_list = ["lineSpacing"]

    if is_title_page:
        # Титульный лист
        if any(pat in text for pat in title_center_patterns):
            p_style["alignment"] = "CENTER"
            p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
            fields_list.extend(["alignment", "indentFirstLine"])
        elif "Студенты" in text or "Преподаватель" in text or "Быстров" in text or "Калинкина" in text or "Кудрявцев" in text or "Бабкин" in text:
            p_style["alignment"] = "START"
            p_style["indentStart"] = {"magnitude": 250, "unit": "PT"}  # блок справа/по центру
            p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
            fields_list.extend(["alignment", "indentStart", "indentFirstLine"])
        else:
            p_style["alignment"] = "CENTER"
            fields_list.append("alignment")

    elif has_image:
        p_style["alignment"] = "CENTER"
        p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
        p_style["spaceAbove"] = {"magnitude": 6, "unit": "PT"}
        p_style["spaceBelow"] = {"magnitude": 2, "unit": "PT"}
        fields_list.extend(["alignment", "indentFirstLine", "spaceAbove", "spaceBelow"])

    elif is_caption:
        p_style["alignment"] = "CENTER"
        p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
        p_style["spaceAbove"] = {"magnitude": 2, "unit": "PT"}
        p_style["spaceBelow"] = {"magnitude": 8, "unit": "PT"}
        fields_list.extend(["alignment", "indentFirstLine", "spaceAbove", "spaceBelow"])

    elif is_formula:
        p_style["alignment"] = "CENTER"
        p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
        p_style["spaceAbove"] = {"magnitude": 6, "unit": "PT"}
        p_style["spaceBelow"] = {"magnitude": 6, "unit": "PT"}
        fields_list.extend(["alignment", "indentFirstLine", "spaceAbove", "spaceBelow"])

    elif is_heading:
        p_style["alignment"] = "START"
        p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
        p_style["spaceAbove"] = {"magnitude": 12, "unit": "PT"}
        p_style["spaceBelow"] = {"magnitude": 6, "unit": "PT"}
        fields_list.extend(["alignment", "indentFirstLine", "spaceAbove", "spaceBelow"])

    elif len(text) > 0:
        # Основной текст по ГОСТ: по ширине, красная строка 1.25 см (36 пт), 1.5 интервал
        p_style["alignment"] = "JUSTIFIED"
        p_style["indentFirstLine"] = {"magnitude": 36, "unit": "PT"}
        p_style["spaceAbove"] = {"magnitude": 0, "unit": "PT"}
        p_style["spaceBelow"] = {"magnitude": 4, "unit": "PT"}
        fields_list.extend(["alignment", "indentFirstLine", "spaceAbove", "spaceBelow"])

    requests.append({
        "updateParagraphStyle": {
            "paragraphStyle": p_style,
            "range": {
                "tabId": tab_id,
                "startIndex": st,
                "endIndex": en
            },
            "fields": ",".join(fields_list)
        }
    })

print(f"Сформировано {len(requests)} запросов на форматирование.")

# Отправка порциями по 50 запросов
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
        print(f"Порция {c_idx // chunk_size + 1}/{(len(requests)-1)//chunk_size + 1} успешно применена: HTTP {resp.status}")

print("\nПолное ГОСТ-оформление вкладки Лаба 6000 успешно завершено!")
