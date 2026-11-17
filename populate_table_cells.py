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
for el in body:
    if "table" in el:
        table_el = el["table"]
        break

if not table_el:
    print("Таблица не найдена!")
    exit(1)

table_data = [
    ["Параметр / Метод", "Однонаправленный ПК", "Взаимный ПК", "Обратимый метод (CD)"],
    ["Канал связи X", "Синхронизируется (k >= 0.45, за 92 с до 10^-14)", "Синхронизируется (k1+k2 >= 0.6, за 101 с при k=0.8)", "Синхронизируется (WT = 2.0 с, за 16 итераций до 1.4e-15)"],
    ["Канал связи Y", "НЕ синхронизируется (lambda_perp = 0, e ~ 2.5)", "НЕ синхронизируется (сохраняется хаос, e ~ 2.5)", "УСПЕШНО синхронизируется (WT = 3.0 с, за 20 итераций до 2.3e-15)"],
    ["Канал связи Z", "Синхронизируется (k >= 0.40, за 286 с при k=0.8)", "Синхронизируется (k1+k2 >= 0.5, за 309 с при k=0.6)", "Синхронизируется (WT = 3.5 с, за 23 итерации до 4.6e-16)"],
    ["Универсальный режим", "Отсутствует", "Отсутствует", "Реализован в точке X*_univ (одновременная сходимость X, Y, Z)"],
    ["Требования к сигналу", "Непрерывный сигнал во всем времени t", "Непрерывный двусторонний сигнал", "Короткий отрезок сигнала WT = 2...3 с"]
]

# Вставляем текст в ячейки с конца таблицы к началу, чтобы индексы не смещались!
insert_requests = []
table_rows = table_el["tableRows"]

for r_idx in range(len(table_rows) - 1, -1, -1):
    row = table_rows[r_idx]
    cells = row["tableCells"]
    for c_idx in range(len(cells) - 1, -1, -1):
        cell = cells[c_idx]
        cell_start = cell["content"][0]["startIndex"]
        text_to_put = table_data[r_idx][c_idx]
        insert_requests.append({
            "insertText": {
                "location": {
                    "tabId": tab_id,
                    "index": cell_start
                },
                "text": text_to_put
            }
        })

b_url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
p_req = urllib.request.Request(
    b_url,
    data=json.dumps({"requests": insert_requests}).encode("utf-8"),
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(p_req) as b_resp:
    print("Текст таблицы успешно заполнен!")
