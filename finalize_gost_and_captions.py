#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт финального ГОСТ-форматирования вкладки "Лаба 6000" (t.76ycwonhwvh):
1. Шрифт: Times New Roman, 14 pt, цвет ТОЛЬКО ЧЕРНЫЙ (#000000)
2. Межстрочный интервал: 1.5 строки (150%)
3. Отсутствие любых границ (borderBottom, borderTop, shading)
4. Выравнивание по ширине (JUSTIFIED) и красная строка 1.25 см (36 pt) для основного текста
5. Заголовки: по левому краю (START), жирные, без отступа
6. Формулы ($$ ... $$), рисунки и подписи (*Рис. N.*): по центру (CENTER)
7. Подписи (*Рис. N.*): курсив (italic), 12 pt
"""

import json
import urllib.request
import re

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

def get_doc():
    url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def batch_update(requests):
    batch_url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
    req = urllib.request.Request(
        batch_url,
        data=json.dumps({"requests": requests}).encode("utf-8"),
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def run():
    doc = get_doc()
    tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
    body = tab["documentTab"]["body"]["content"]
    doc_end = body[-1]["endIndex"]

    print(f"Всего элементов: {len(body)}, диапазон: 1 - {doc_end}")
    requests = []

    # 1. Глобальный базовый стиль текста (Times New Roman, 14 pt, Черный #000000)
    requests.append({
        "updateTextStyle": {
            "textStyle": {
                "weightedFontFamily": {"fontFamily": "Times New Roman"},
                "fontSize": {"magnitude": 14, "unit": "PT"},
                "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}},
                "backgroundColor": {}
            },
            "range": {
                "tabId": tab_id,
                "startIndex": 1,
                "endIndex": doc_end - 1
            },
            "fields": "weightedFontFamily,fontSize,foregroundColor,backgroundColor"
        }
    })

    # Регулярные выражения для истинных заголовков
    heading_re = re.compile(r"^(#+\s+|[1-6]\.\s+[А-ЯЁ]|[1-6]\.[1-5]\.\s+[А-ЯЁ]|[1-6]\.[1-5]\.[1-5]\.\s+[А-ЯЁ]|Сводные метрологические|Выводы|Заключение|Список литературы)")

    for elem in body:
        if "paragraph" in elem:
            p = elem["paragraph"]
            st = elem.get("startIndex")
            en = elem.get("endIndex")
            if st is None or en is None or st >= en:
                continue

            text = "".join(el.get("textRun", {}).get("content", "") for el in p.get("elements", []))
            clean_t = text.strip()
            has_objs = any("inlineObjectElement" in el for el in p.get("elements", []))

            # Очистка границ и заливок по умолчанию для всех
            p_style = {
                "lineSpacing": 150
            }
            fields = "lineSpacing,borderBottom,borderTop,borderBetween,borderLeft,borderRight,shading"

            # Титульный лист (до индекса 432)
            if en <= 432:
                p_style["alignment"] = "CENTER"
                p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
                fields += ",alignment,indentFirstLine"
                requests.append({
                    "updateParagraphStyle": {
                        "paragraphStyle": p_style,
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": fields
                    }
                })
                continue

            # Рисунок (абзац содержит картинку)
            if has_objs:
                p_style["alignment"] = "CENTER"
                p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
                fields += ",alignment,indentFirstLine"
                requests.append({
                    "updateParagraphStyle": {
                        "paragraphStyle": p_style,
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": fields
                    }
                })
                continue

            # Подпись к рисунку (*Рис. N.*)
            if clean_t.startswith("*Рис.") or clean_t.startswith("Рис."):
                p_style["alignment"] = "CENTER"
                p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
                fields += ",alignment,indentFirstLine"
                requests.append({
                    "updateParagraphStyle": {
                        "paragraphStyle": p_style,
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": fields
                    }
                })
                # Шрифт подписи: 12 pt, курсив, черный
                requests.append({
                    "updateTextStyle": {
                        "textStyle": {
                            "fontSize": {"magnitude": 12, "unit": "PT"},
                            "italic": True,
                            "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}}
                        },
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": "fontSize,italic,foregroundColor"
                    }
                })
                continue

            # Формула на отдельной строке ($$ ... $$)
            if clean_t.startswith("$$") or "\\begin{" in clean_t:
                p_style["alignment"] = "CENTER"
                p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
                fields += ",alignment,indentFirstLine"
                requests.append({
                    "updateParagraphStyle": {
                        "paragraphStyle": p_style,
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": fields
                    }
                })
                continue

            # Истинный заголовок
            if heading_re.match(clean_t) and len(clean_t) < 140:
                p_style["alignment"] = "START"
                p_style["indentFirstLine"] = {"magnitude": 0, "unit": "PT"}
                fields += ",alignment,indentFirstLine"
                requests.append({
                    "updateParagraphStyle": {
                        "paragraphStyle": p_style,
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": fields
                    }
                })
                # Жирный заголовок
                requests.append({
                    "updateTextStyle": {
                        "textStyle": {
                            "bold": True,
                            "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}}
                        },
                        "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                        "fields": "bold,foregroundColor"
                    }
                })
                continue

            # Обычный абзац текста
            p_style["alignment"] = "JUSTIFIED"
            p_style["indentFirstLine"] = {"magnitude": 36, "unit": "PT"}  # 1.25 см
            fields += ",alignment,indentFirstLine"
            requests.append({
                "updateParagraphStyle": {
                    "paragraphStyle": p_style,
                    "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                    "fields": fields
                }
            })
            requests.append({
                "updateTextStyle": {
                    "textStyle": {
                        "bold": False,
                        "italic": False,
                        "fontSize": {"magnitude": 14, "unit": "PT"},
                        "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}}
                    },
                    "range": {"tabId": tab_id, "startIndex": st, "endIndex": en},
                    "fields": "bold,italic,fontSize,foregroundColor"
                }
            })

    print(f"Отправка {len(requests)} запросов форматирования по ГОСТу...")
    res = batch_update(requests)
    print("Форматирование успешно применено!")

if __name__ == "__main__":
    run()
