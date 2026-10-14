# 指标回放验证（8 真解 + 7 假解）

- 准入线：C_s02≥0.83, PCC0≥0.82, PCC2≥0.83
- 警示线：rho≥0.2 或 C_min≤0.3（32/56px）

## 真解回放

| 图 | C_s02 | 准入 | 局部警示 |
|---|---|---|---|
| 4Singularities | 0.8931 | ✓ | C_min_32px=0.15 |
| 8Singularities(1) | 0.9453 | ✓ | 无 |
| 5Singularities | 0.9355 | ✓ | 无 |
| 6Singularities | 0.9362 | ✓ | 无 |
| 8Singularities(2) | 0.8457 | ✓ | C_min_32px=-0.17 |
| 9Singularities | 0.8526 | ✓ | C_min_32px=-0.18 |
| 11Singularities | 0.8351 | ✓ | rho_32px=0.24；C_min_32px=-0.17 |
| 10Singularities | 0.8629 | ✓ | 无 |

## 假解回放

| 来源 | C_s02 | 准入 | 局部警示 |
|---|---|---|---|
| 8Sing1 two-mode champion | 0.9255 | 放行 | C_min_32px=-0.07 |
| 5Sing runner-up (1 core) | 0.9353 | 放行 | 无 |
| 10Sing two-mode | 0.8176 | **拦截** | rho_32px=0.43；C_min_32px=-0.21；rho_56px=0.80 |
| 11Sing two-mode | 0.8361 | 放行 | rho_32px=0.33；C_min_32px=-0.53；rho_56px=0.80 |
| 9Sing three-mode top (12 cores) | 0.8781 | 放行 | C_min_32px=-0.10 |
| 10Sing three-mode close (12 cores) | 0.8231 | **拦截** | C_min_32px=-0.12 |
| 10Sing four-mode highest-corr (8 cores) | 0.8619 | 放行 | rho_32px=0.29；C_min_32px=0.15；rho_56px=0.40 |

## 结论

- 真解：8/8 通过准入；4 个带局部警示（低分真解局部失配，符合预期）；
- 假解：2/7 被准入线拦截；6 个带局部警示；
- 警示率差异 + 结构验收协作 = 完整防线（全局分放行的假解由结构验收终审拦下）。