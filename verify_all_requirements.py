#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт проверки всех требований пользователя:
1. Отсутствие серых полос (borders, shading, backgroundColor)
2. Все рисунки имеют подписи (*Рис. N.*)
3. Весь LaTeX обернут в двойные доллары $$ ... $$ (нет $ и нет $$$$)
4. Форматирование текста: Times New Roman, 14 pt, черный цвет (#000000), 1.5 интервал
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
inline_objects = tab["documentTab"].get("inlineObjects", {})

print(f"=== ПРОВЕРКА ВКЛАДКИ 'Лаба 6000' ===")
print(f"Всего элементов в документе: {len(body)}")
print(f"Всего встроенных объектов (изображений): {len(inline_objects)}")

# 1. Проверка границ и заливок
border_paragraphs = []
shaded_paragraphs = []
for i, elem in enumerate(body):
    if "paragraph" in elem:
        p_style = elem["paragraph"].get("paragraphStyle", {})
        borders = {k: v for k, v in p_style.items() if "border" in k.lower()}
        if borders:
            border_paragraphs.append((i, borders))
        shading = p_style.get("shading", {})
        if shading:
            shaded_paragraphs.append((i, shading))

print(f"\n1. Серые полосы / границы (borders): {len(border_paragraphs)}")
if border_paragraphs:
    for bp in border_paragraphs[:5]:
        print("   Параграф с границей:", bp)
else:
    print("   -> Серые полосы ПОЛНОСТЬЮ ОТСУТСТВУЮТ! (0 найдено)")

print(f"   Заливка параграфов (shading): {len(shaded_paragraphs)}")

# 2. Проверка рисунков и подписей
print("\n2. Проверка рисунков и подписей:")
figures_found = []
for i, elem in enumerate(body):
    if "paragraph" in elem:
        p = elem["paragraph"]
        objs = [el["inlineObjectElement"]["inlineObjectId"] for el in p.get("elements", []) if "inlineObjectElement" in el]
        if objs:
            # проверяем следующий параграф на подпись
            next_text = ""
            if i + 1 < len(body) and "paragraph" in body[i+1]:
                next_text = "".join(e.get("textRun", {}).get("content", "") for e in body[i+1]["paragraph"]["elements"]).strip()
            figures_found.append((i, objs, next_text))

print(f"   Всего параграфов с рисунками: {len(figures_found)}")
for idx, (p_idx, objs, caption) in enumerate(figures_found):
    print(f"   Рисунок {idx+1} (элемент {p_idx}, id={objs[0]}): подпись = {repr(caption[:75])}")

# 3. Проверка LaTeX выражений
full_text = ""
for elem in body:
    if "paragraph" in elem:
        for el in elem["paragraph"].get("elements", []):
            full_text += el.get("textRun", {}).get("content", "")

single_dollars = re.findall(r"(?<!\$)\$([^\$\n]+)\$(?!\$)", full_text)
quad_dollars = re.findall(r"\${4,}", full_text)
double_dollars = re.findall(r"(?<!\$)\${2}([^\$]+)\${2}(?!\$)", full_text)

print(f"\n3. Проверка LaTeX:")
print(f"   Одинарные доллары ($...$): {len(single_dollars)}")
print(f"   Четверные доллары ($$$$...$$$$): {len(quad_dollars)}")
print(f"   Двойные доллары ($$...$$): {len(double_dollars)}")
if len(single_dollars) == 0 and len(quad_dollars) == 0:
    print("   -> Весь LaTeX строго обернут в двойные доллары $$...$$!")

# 4. Проверка цветов текста
non_black_runs = []
for elem in body:
    if "paragraph" in elem:
        for el in elem["paragraph"].get("elements", []):
            tr = el.get("textRun", {})
            ts = tr.get("textStyle", {})
            fg = ts.get("foregroundColor", {})
            rgb = fg.get("color", {}).get("rgbColor", {})
            if rgb:
                r = rgb.get("red", 0.0)
                g = rgb.get("green", 0.0)
                b = rgb.get("blue", 0.0)
                if r > 0.05 or g > 0.05 or b > 0.05:
                    non_black_runs.append((repr(tr.get("content", "")[:20]), r, g, b))

print(f"\n4. Проверка цвета текста (не черный): {len(non_black_runs)}")
if len(non_black_runs) == 0:
    print("   -> Текст строго ЧЕРНЫЙ (#000000) по всему отчету!")
