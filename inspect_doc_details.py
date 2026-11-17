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

print("=== CHECKING IMAGES AND CAPTIONS ===")
for i, elem in enumerate(body):
    if "paragraph" in elem:
        p = elem["paragraph"]
        for el in p.get("elements", []):
            if "inlineObjectElement" in el:
                obj_id = el["inlineObjectElement"]["inlineObjectId"]
                st = elem.get("startIndex")
                en = elem.get("endIndex")
                next_text = ""
                if i + 1 < len(body) and "paragraph" in body[i+1]:
                    next_text = "".join(e.get("textRun", {}).get("content", "") for e in body[i+1]["paragraph"]["elements"]).strip()
                print(f"Image {obj_id} at elem {i} (chars {st}-{en}). Next elem {i+1} text: {repr(next_text[:100])}")

print("\n=== CHECKING LATEX IN TEXT ===")
full_text = ""
for elem in body:
    if "paragraph" in elem:
        for el in elem["paragraph"].get("elements", []):
            full_text += el.get("textRun", {}).get("content", "")

single_dollars = re.findall(r"(?<!\$)\$([^\$\n]+)\$(?!\$)", full_text)
print(f"Single dollar LaTeX count: {len(single_dollars)}")
for s in single_dollars[:10]:
    print(f"  Single: ${s}$")

quad_dollars = re.findall(r"\${4,}([^\$]+)\${4,}", full_text)
print(f"Quadruple dollar LaTeX count: {len(quad_dollars)}")
for q in quad_dollars[:10]:
    print(f"  Quad: {q[:40]}")

triple_dollars = re.findall(r"(?<!\$)\${3}([^\$]+)\${3}(?!\$)", full_text)
print(f"Triple dollar LaTeX count: {len(triple_dollars)}")
for t in triple_dollars[:10]:
    print(f"  Triple: {t[:40]}")

double_dollars = re.findall(r"(?<!\$)\${2}([^\$]+)\${2}(?!\$)", full_text)
print(f"Double dollar LaTeX count: {len(double_dollars)}")
