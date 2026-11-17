import json
import urllib.request

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"

url = "https://docs.googleapis.com/v1/documents/" + doc_id + "?includeTabsContent=true"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + access_token})
with urllib.request.urlopen(req) as resp:
    doc = json.loads(resp.read().decode())

tab_51 = None
for tab in doc.get("tabs", []):
    if tab.get("tabProperties", {}).get("tabId") == "t.76ycwonhwvh":
        tab_51 = tab
        break

if tab_51:
    body = tab_51.get("documentTab", {}).get("body", {})
    elements = body.get("content", [])
    print("Tab 5.1 found! Total elements:", len(elements))
    for i, el in enumerate(elements[:45]):
        p = el.get("paragraph")
        if p:
            text = "".join(tr.get("textRun", {}).get("content", "") for tr in p.get("elements", []))
            style = p.get("paragraphStyle", {}).get("namedStyleType", "")
            align = p.get("paragraphStyle", {}).get("alignment", "DEFAULT")
            st = el.get("startIndex")
            en = el.get("endIndex")
            print(f"[{i}] {st}..{en} ({style}, {align}): {repr(text.strip()[:60])}")
        elif "table" in el:
            print(f"[{i}] TABLE")
