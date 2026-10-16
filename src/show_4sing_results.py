# -*- coding: utf-8 -*-
"""show_4sing_results.py — 打印 4奇点搜索 top 详情"""
import json
from pathlib import Path

_R = Path(__file__).resolve().parent
p = (_R / '归档_八图分解' / '纯基调整_2026-10-16'
     / '第2层_搜索' / '4奇点' / '搜索结果.json')
d = json.load(open(p, encoding='utf-8'))
for track in ('HG', 'LG'):
    print(f'=== 纯{track} top5 (any) ===')
    for b in d[track]['best_any']:
        print(f"  C={b['C_s02']:.4f} struct={b['struct']} "
              f"cores={b['cores']} mean_nn={b['mean_nn']:.1f} "
              f"combo={'+'.join(b['combo'])}")
        print(f"    theta={[round(x, 3) for x in b['theta']]}")
        print(f"    sx={b['sx']} sy={b['sy']} rot={b['rot']} "
              f"shift={[round(x, 2) for x in b['shift']]}")
