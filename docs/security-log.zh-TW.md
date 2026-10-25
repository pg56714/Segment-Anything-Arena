# 安全性紀錄

[English](security-log.md) | 繁體中文

- 日期: 2026-10-25
- 分支: `fix/security-upgrades`
- 範圍: `requirements.txt`

## 摘要

gradio 從 4.19.2 升到已修補的版本。pillow 因為 gradio 限制版本，仍有未修正的漏洞。

## 修改內容

| 項目 | 修改前 | 修改後 | 原因 |
|---|---|---|---|
| gradio | 4.19.2 | >=6.16 (6.30.0) | 任意檔案讀取、SSRF、路徑穿越漏洞（6.16 修正） |

## 使用前須知

- 原本的註解提到 gradio 4.19.2 執行較快，6.x 的介面可能感覺較慢。

## 尚未解決的問題

### 被上游限制的套件版本

| 套件 | 版本 | 漏洞 | 無法修正的原因 | 影響與因應 |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 筆（13.0.0 修正） | gradio 6.16 以上要求 `pillow<13`，目前沒有允許 Pillow 13 的 gradio 版本。 | 解碼惡意製作的圖片（例如 TIFF、PDF）可能讓程式當掉或越界讀取記憶體。請只在本機執行 Gradio 介面（不要開 `share=True`、不要對外開放連接埠），只開啟可信任的圖片；等 gradio 允許 Pillow 13 後再升級。 |

## 驗證方式

- 以 osv-scanner 掃描解析後的 `requirements.txt`，只剩 pillow。
- 把 EfficientSAM、EfficientViT-SAM 模型換成假模組後，介面在 gradio 6.30.0 下能正常建立並提供服務。
- `annotate_image_with_box_prompt_result()` 在 supervision 0.30.9 下可正常執行。

## 未驗證項目

- 用 GPU 與實際權重執行推論（測試環境沒有 GPU 與權重檔）。
