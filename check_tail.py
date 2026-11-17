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

tab = [t for t in doc["tabs"] if t["tabProperties"]["tabId"] == "t.76ycwonhwvh"][0]
content = tab["documentTab"]["body"]["content"]
print("Total elements:", len(content))
for i in range(145, len(content)):
    el = content[i]
    p = el.get("paragraph")
    if p:
        text = "".join(tr.get("textRun", {}).get("content", "") for tr in p.get("elements", []))
        st = el.get("startIndex")
        en = el.get("endIndex")
        print(f"[{i}] {st}..{en}: {repr(text.strip()[:80])}")
