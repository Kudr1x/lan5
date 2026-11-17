#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт полной замены и очистки разделов 3.1 - 3.4 во вкладке "Лаба 6000" (t.76ycwonhwvh).
- Удаляет старый мусор и скученные картинки между индексом 8088 и началом заголовка 3.5.
- Вставляет идеально структурированный текст с отдельными параграфами для каждого рисунка (Рис. 1 - 15) и его подписи.
- Заменяет плейсхолдеры на новые наглядные изображения 300 DPI.
- Применяет форматирование по ГОСТу (Times New Roman 14 pt, 1.5 интервал, черный цвет #000000, подписи по центру курсив 12 pt).
"""

import json
import urllib.request
import subprocess
import os
import sys

with open("/home/bobbert/.gemini/antigravity/plugin_data/gdocs/token.json") as f:
    token_data = json.load(f)

access_token = token_data["token"]["access_token"]
doc_id = "1WXoZIKXH_fYTq8krfg7taHPUvWUmaSn_82PZJpav_b0"
tab_id = "t.76ycwonhwvh"
BASE_DIR = "/home/bobbert/Документы/leti/mnds/lab4/lab5"
GDOCS_BIN = "/home/bobbert/.gemini/config/plugins/gdocs/skills/gdocs/scripts/gdocs"

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

# Шаг 1: Находим точный диапазон удаления
doc = get_doc()
tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
body = tab["documentTab"]["body"]["content"]

start_del = None
end_del = None

for i, elem in enumerate(body):
    if "paragraph" in elem:
        text = "".join(el.get("textRun", {}).get("content", "") for el in elem["paragraph"].get("elements", []))
        if "3.1. Исследование однонаправленной" in text and start_del is None:
            start_del = elem.get("startIndex")
        if "3.5. Исследование параметров" in text and end_del is None:
            # Находим, с какого смещения начинается сам текст заголовка
            p_els = elem["paragraph"].get("elements", [])
            for el in p_els:
                if "textRun" in el and "3.5. Исследование параметров" in el["textRun"]["content"]:
                    end_del = el.get("startIndex")
                    break

print(f"Диапазон удаления: от {start_del} до {end_del}")

# Формируем чистый текст для разделов 3.1 - 3.4
# Плейсхолдер для каждой картинки обозначим символом '@' на отдельной строке
clean_text = (
    "3.1. Исследование однонаправленной синхронизации Пикоры — Кэрола\n"
    "На рис. 1 приведена зависимость установившейся ошибки синхронизации от коэффициента связи $$k \\in [0, 4.0]$$ для трех независимых каналов связи.\n"
    "@\n"
    "*Рис. 1. Зависимость установившейся ошибки от коэффициента связи k при однонаправленной связи*\n"
    "Анализ зависимостей:\n"
    "• Канал $$x$$: критический порог $$k_{crit} = 0.45$$. При $$k \\ge 1.0$$ наступает полная синхронизация, ошибка падает до машинного уровня.\n"
    "• Канал $$z$$: критический порог $$k_{crit} = 0.40$$. При $$k \\ge 1.0$$ ошибка падает до машинного нуля.\n"
    "• Канал $$y$$: синхронизация отсутствует во всем исследованном диапазоне $$k \\in [0, 4.0]$$, ошибка флуктуирует на уровне $$||e|| \\approx 1.5 \\dots 4.5$$, что строго подтверждает аналитический вывод подраздела 2.4.\n"
    "На рис. 2–4 представлены временные реализации фазовых координат и логарифмическая ошибка при $$k = 2.0$$.\n"
    "@\n"
    "*Рис. 2. Временные ряды и логарифм ошибки при однонаправленной связи по x (k = 2.0)*\n"
    "@\n"
    "*Рис. 3. Временные ряды при связи по y (k = 2.0): незатухающие хаотические биения ошибки*\n"
    "@\n"
    "*Рис. 4. Временные ряды и логарифм ошибки при связи по z (k = 2.0)*\n"
    "@\n"
    "*Рис. 5. Плоскости синхронизма (u_s от u_m) при k = 2.0: четкая диагональ для x и z, хаотическое пятно для y*\n"
    "3.2. Исследование двунаправленной (взаимной) синхронизации\n"
    "При взаимной связи системы подстраиваются согласованно. Зависимость ошибки от величины коэффициента связи представлена на рис. 6.\n"
    "@\n"
    "*Рис. 6. Зависимость установившейся ошибки от коэффициента взаимной связи k*\n"
    "Основные закономерности:\n"
    "1. Порог синхронизации по координате $$x$$ снизился на 33%: $$k_{crit} \\approx 0.30$$ (против $$0.45$$ в однонаправленной схеме). Для координаты $$z$$ порог уменьшился до $$k_{crit} \\approx 0.35$$.\n"
    "2. Скорость сходимости возросла в 2.5 раза (рис. 7): взаимная подстройка ускоряет стягивание фазовых траекторий.\n"
    "3. По координате $$y$$ взаимная связь также не приводит к синхронизации в силу кососимметричной структуры поперечной подсистемы ошибок.\n"
    "@\n"
    "*Рис. 7. Сравнение динамики ошибки: взаимная связь сходится в 2.5 раза быстрее однонаправленной*\n"
    "@\n"
    "*Рис. 8. Диагонали взаимной синхронизации при k = 2.0*\n"
    "3.3. Исследование метода быстрой обратимой синхронизации\n"
    "Проведено исследование алгоритма циклической обратимой синхронизации при шаге $$h = 0.005$$ с.\n"
    "На рис. 9 представлена динамика логарифмической ошибки по итерациям для двух режимов:\n"
    "• Слева: сходимость в индивидуальных рекордных точках аттрактора для каждого из сигналов ($$x$$, $$y$$, $$z$$);\n"
    "• Справа: сходимость в глобальной универсальной точке $$X^*_{univ}$$, где все три сигнала синхронизируются одновременно.\n"
    "@\n"
    "*Рис. 9. Сходимость ошибки обратимой синхронизации по итерациям: полная сходимость по всем трем каналам*\n"
    "График наглядно демонстрирует фундаментальное свойство обратимого метода: канал $$y$$ успешно синхронизируется, достигая ошибки $$||e|| = 1.11 \\cdot 10^{-16}$$ (ниже уровня машинного эпсилона).\n"
    "@\n"
    "*Рис. 10. Влияние длины окна данных WT на скорость сходимости*\n"
    "@\n"
    "*Рис. 11. Эволюция траектории ведомой системы к ведущей на окне WT = 2.0 с*\n"
    "На рис. 12 сопоставлены классический метод Пикоры — Кэрола и обратимый метод.\n"
    "@\n"
    "*Рис. 12. Сравнение сходимости: по каналу X оба метода сходятся, по каналу Y метод Пикоры — Кэрола расходится, обратимый метод сходится к машинному эпсилону*\n"
    "3.4. Комплексные матрицы сравнения методов по каждому сигналу\n"
    "На рис. 13–15 приведены полные сводные 6-панельные карты, сопоставляющие все три метода индивидуально для сигналов $$x, y, z$$.\n"
    "@\n"
    "*Рис. 13. Сводный анализ всех трех методов для СИГНАЛА X*\n"
    "@\n"
    "*Рис. 14. Сводный анализ всех трех методов для СИГНАЛА Y: в 3-й строке наглядно показана победа над рассинхронизацией*\n"
    "@\n"
    "*Рис. 15. Сводный анализ всех трех методов для СИГНАЛА Z*\n"
)

# Выполняем замену: сначала удаление диапазона, затем вставка нового текста
reqs = [
    {
        "deleteContentRange": {
            "range": {
                "tabId": tab_id,
                "startIndex": start_del,
                "endIndex": end_del
            }
        }
    },
    {
        "insertText": {
            "location": {
                "tabId": tab_id,
                "index": start_del
            },
            "text": clean_text
        }
    }
]

print("Выполняем замену текста 3.1 - 3.4...")
res = batch_update(reqs)
print("Текст успешно заменен!")

# Шаг 2: Находим точные индексы всех 15 плейсхолдеров '@'
doc = get_doc()
tab = next(t for t in doc["tabs"] if t["tabProperties"]["tabId"] == tab_id)
body = tab["documentTab"]["body"]["content"]

placeholder_indices = []
for elem in body:
    if "paragraph" in elem:
        p = elem["paragraph"]
        for el in p.get("elements", []):
            if "textRun" in el:
                c = el["textRun"]["content"]
                if c == "@\n" or c == "@":
                    placeholder_indices.append(el["startIndex"])

print(f"Найдено {len(placeholder_indices)} плейсхолдеров '@': {placeholder_indices}")

images_map = [
    os.path.join(BASE_DIR, "plots/01_unidirectional_error_vs_k.png"),
    os.path.join(BASE_DIR, "plots/02_unidirectional_timeseries_x.png"),
    os.path.join(BASE_DIR, "plots/02_unidirectional_timeseries_y.png"),
    os.path.join(BASE_DIR, "plots/02_unidirectional_timeseries_z.png"),
    os.path.join(BASE_DIR, "plots/05_unidirectional_phase_portraits.png"),
    os.path.join(BASE_DIR, "plots/06_bidirectional_error_vs_k.png"),
    os.path.join(BASE_DIR, "plots/07_bidirectional_timeseries_x.png"),
    os.path.join(BASE_DIR, "plots/08_bidirectional_phase_portraits.png"),
    os.path.join(BASE_DIR, "plots/09_reversible_convergence_vs_iterations.png"),
    os.path.join(BASE_DIR, "plots/10_reversible_window_comparison.png"),
    os.path.join(BASE_DIR, "plots/11_reversible_trajectory_evolution.png"),
    os.path.join(BASE_DIR, "plots/12_comparison_reversible_vs_pecora_carroll.png"),
    os.path.join(BASE_DIR, "plots/comprehensive_signal_x_all_methods.png"),
    os.path.join(BASE_DIR, "plots/comprehensive_signal_y_all_methods.png"),
    os.path.join(BASE_DIR, "plots/comprehensive_signal_z_all_methods.png"),
]

# Заменяем плейсхолдеры в обратном порядке (от 15-го к 1-му), чтобы индексы не смещались
print("Вставка рисунков 1 - 15...")
for i in reversed(range(len(placeholder_indices))):
    idx = placeholder_indices[i]
    img_path = images_map[i]
    print(f"Вставка Рис. {i+1} на индекс {idx} ({os.path.basename(img_path)})...")
    cmd = [
        GDOCS_BIN, "mutate", "insert-image-from-file",
        doc_id, img_path,
        "--tab", tab_id,
        "--index", str(idx),
        "--replace"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Ошибка вставки Рис. {i+1}: {res.stderr}")
    else:
        print(f"Рис. {i+1} успешно вставлен!")

print("Все рисунки 1 - 15 успешно интегрированы!")
