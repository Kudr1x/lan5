import json
import urllib.request
import os
import subprocess

with open('/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json') as f:
    token_data = json.load(f)
token = token_data['token']['access_token']
doc_id = '1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0'
tab_id = 't.ru0l3r4wrur8'

url = f'https://docs.googleapis.com/v1/documents/{doc_id}?includeTabsContent=true'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
with urllib.request.urlopen(req) as resp:
    d = json.loads(resp.read().decode())
    tab51 = [t for t in d['tabs'] if t['tabProperties']['tabId'] == tab_id][0]
    body = tab51['documentTab']['body']['content']
    
    start_del = None
    end_del = None
    for el in body:
        p = el.get('paragraph')
        if p:
            t = ''.join([e.get('textRun', {}).get('content', '') for e in p.get('elements', [])])
            if 'Параметр / Метод |' in t and start_del is None:
                start_del = el.get('startIndex')
            if 'Короткий отрезок сигнала' in t:
                end_del = el.get('endIndex')

print('start_del:', start_del, 'end_del:', end_del)

if start_del and end_del:
    del_req = {
        'deleteContentRange': {
            'range': {
                'tabId': tab_id,
                'startIndex': start_del,
                'endIndex': end_del
            }
        }
    }
    ins_tbl_req = {
        'insertTable': {
            'rows': 6,
            'columns': 4,
            'location': {
                'tabId': tab_id,
                'index': start_del
            }
        }
    }
    b_url = f'https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate'
    p_req = urllib.request.Request(b_url, data=json.dumps({'requests': [del_req, ins_tbl_req]}).encode('utf-8'),
                                   headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
                                   method='POST')
    with urllib.request.urlopen(p_req) as b_resp:
        print('Table inserted successfully!')
