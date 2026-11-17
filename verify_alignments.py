import json
import urllib.request

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"

url = f"https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab = [t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id][0]
content = tab["documentTab"]["body"]["content"]

counts = {"JUSTIFIED": 0, "CENTER": 0, "START": 0, "DEFAULT": 0, "OTHER": 0}
formula_count = 0
formula_centered = 0

for i, el in enumerate(content):
    p = el.get("paragraph")
    if not p:
        continue
    style = p.get("paragraphStyle", {})
    align = style.get("alignment", "DEFAULT")
    counts[align] = counts.get(align, 0) + 1
    text = "".join(tr.get("textRun", {}).get("content", "") for tr in p.get("elements", [])).strip()
    if "$$" in text:
        formula_count += 1
        if align == "CENTER":
            formula_centered += 1
        else:
            print(f"Formula not centered [{i}] ({align}): {repr(text[:50])}")

print("Alignment distribution across paragraphs:", counts)
print(f"Total LaTeX display formula paragraphs: {formula_count}, Centered: {formula_centered}")
