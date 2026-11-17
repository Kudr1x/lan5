#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import urllib.request

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.ru0l3r4wrur8"

url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    d = json.loads(resp.read().decode())
    tab51 = [t for t in d["tabs"] if t["tabProperties"]["tabId"] == tab_id][0]
    body = tab51["documentTab"]["body"]["content"]

table_el = None
tbl_start = None
tbl_end = None
for el in body:
    if "table" in el:
        table_el = el["table"]
        tbl_start = el["startIndex"]
        tbl_end = el["endIndex"]
        break

reqs = []

# Форматирование текста таблицы: Times New Roman, 11 pt, строго черный
reqs.append({
    "updateTextStyle": {
        "range": {
            "tabId": tab_id,
            "startIndex": tbl_start,
            "endIndex": tbl_end
        },
        "textStyle": {
            "weightedFontFamily": {"fontFamily": "Times New Roman"},
            "fontSize": {"magnitude": 11, "unit": "PT"},
            "foregroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}}
        },
        "fields": "weightedFontFamily,fontSize,foregroundColor"
    }
})

# Заголовочная строка: жирный шрифт, легкая заливка
header_row = table_el["tableRows"][0]
h_start = header_row["tableCells"][0]["startIndex"]
h_end = header_row["tableCells"][-1]["endIndex"]

reqs.append({
    "updateTextStyle": {
        "range": {
            "tabId": tab_id,
            "startIndex": h_start,
            "endIndex": h_end
        },
        "textStyle": {
            "bold": True
        },
        "fields": "bold"
    }
})

# Заливка шапки таблицы
reqs.append({
    "updateTableCellStyle": {
        "tableRange": {
            "tableCellLocation": {
                "tableStartLocation": {"tabId": tab_id, "index": tbl_start},
                "rowIndex": 0,
                "columnIndex": 0
            },
            "rowSpan": 1,
            "columnSpan": 4
        },
        "tableCellStyle": {
            "backgroundColor": {
                "color": {"rgbColor": {"red": 0.93, "green": 0.93, "blue": 0.93}}
            }
        },
        "fields": "backgroundColor"
    }
})

# Первая колонка таблицы: полужирный шрифт
for r_idx in range(1, len(table_el["tableRows"])):
    first_cell = table_el["tableRows"][r_idx]["tableCells"][0]
    c_start = first_cell["startIndex"]
    c_end = first_cell["endIndex"]
    reqs.append({
        "updateTextStyle": {
            "range": {
                "tabId": tab_id,
                "startIndex": c_start,
                "endIndex": c_end
            },
            "textStyle": {
                "bold": True
            },
            "fields": "bold"
        }
    })

b_url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
p_req = urllib.request.Request(
    b_url,
    data=json.dumps({"requests": reqs}).encode("utf-8"),
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(p_req) as b_resp:
    print("Стилизация таблицы успешно завершена!")
