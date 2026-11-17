#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import urllib.request
import re

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

# 1. Fetch document structure
url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab = None
for t in doc.get("tabs", []):
    if t.get("tabProperties", {}).get("tabId") == tab_id:
        tab = t
        break

if not tab:
    print("Tab not found!")
    exit(1)

content = tab.get("documentTab", {}).get("body", {}).get("content", [])

print(f"Total structural elements: {len(content)}")

title_patterns = [
    "МИНОБРНАУКИ РОССИИ",
    "Санкт-Петербургский государственный",
    "электротехнический университет",
    "«ЛЭТИ» им. В.И. Ульянова",
    "Кафедра систем автоматизированного проектирования",
    "Отчет",
    "по лабораторной работе №5",
    "по дисциплине «Моделирование нелинейных динамических систем»",
    "Тема: «Исследование непрерывной и быстрой",
    "Санкт-Петербург",
    "2026"
]

requests = []

for i, el in enumerate(content):
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

    # Determine desired alignment
    target_alignment = None

    if any(pat in text for pat in title_patterns) and st < 700:
        target_alignment = "CENTER"
    elif has_image:
        target_alignment = "CENTER"
    elif text.startswith("*Рис.") or text.startswith("Рис."):
        target_alignment = "CENTER"
    elif "$$" in text or "\\begin{" in text or "\\dfrac{" in text or "\\text{div}" in text or "\\Phi" in text:
        target_alignment = "CENTER"
    elif named_style.startswith("HEADING_"):
        target_alignment = "START"
    elif len(text) > 0:
        # Body text
        target_alignment = "JUSTIFIED"

    if target_alignment:
        requests.append({
            "updateParagraphStyle": {
                "paragraphStyle": {
                    "alignment": target_alignment
                },
                "range": {
                    "tabId": tab_id,
                    "startIndex": st,
                    "endIndex": en
                },
                "fields": "alignment"
            }
        })
        print(f"Elem [{i}] ({st}..{en}) -> {target_alignment}: {repr(text[:45])}")

print(f"\nGenerated {len(requests)} alignment requests.")

# Send batchUpdate in chunks of 50 requests
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
        print(f"Chunk {c_idx // chunk_size + 1} applied: HTTP {resp.status}")

print("Все стили выравнивания успешно применены!")
