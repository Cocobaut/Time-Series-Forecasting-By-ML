"""
TỰ ĐỘNG XUẤT CÁC BẢNG ĐỐI SÁNH TRỌNG TÂM (MARKDOWN + LATEX)

Tổng hợp kết quả JSON do các kịch bản thực nghiệm sinh ra — KHÔNG có số liệu nào được ghi tay:
  BẢNG 1      Benchmark -> Linear -> ML Baseline -> P1 -> P2 -> P3 (mean ± std qua các seed)
              <- results/07_improvement/a_p1_p3/improvement_multiseed.json, results/01_baseline/linear_results.json
  BẢNG 2      Hướng 2: hòa giải phân cấp (ML Baseline)    <- results/07_improvement/b_hierarchical/hierarchical_results.json
  BẢNG 3      Hướng 3: P3 + hòa giải phân cấp             <- results/07_improvement/c_combined_p3_hierarchical/combined_results.json
  Bảng phụ A  Rolling Origin CV                           <- results/04_time_series_cv/cv_*.json
  Bảng phụ B  Multi-seed + chi phí tính toán              <- results/05_multiseed/multiseed_summary.json
  Bảng phụ C  Tuning ngân sách đồng nhất                  <- results/03_tuning/tuning_comparison.json
  Bảng phụ D  Độ nhạy MinT theo nguồn phần dư             <- *_resval.json của Hướng 2 và Hướng 3
  Bảng phụ E  Ablation + feature importance               <- results/06_ablation/ablation_results.json
  Cấu hình phần cứng                                      <- results/hardware_info.json

Đầu ra: results/08_benchmark_tables/benchmark_tables.md, benchmark_tables.tex
Tùy chọn --update-docs: chèn bảng Markdown vào Final_Analyze.md giữa hai marker
    <!-- BENCHMARK_TABLES:START --> ... <!-- BENCHMARK_TABLES:END -->

Cách chạy:
    python Src/Evaluation/generate_benchmark_tables.py
    python Src/Evaluation/generate_benchmark_tables.py --update-docs
    python Src/Evaluation/generate_benchmark_tables.py --quick      # đọc kết quả chạy thử *_quick
"""
import sys
import argparse
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import get_result_dir
from Src.Experiments.common import HARDWARE_INFO_PATH, setup_console, load_json, markdown_table

RESULT_DIR = get_result_dir("benchmark")
CV_DIR = get_result_dir("cv")
HIER_DIR = get_result_dir("hierarchical")      # Hướng 2
COMB_DIR = get_result_dir("combined")          # Hướng 3
DOC_PATH = BASE_DIR / "Final_Analyze.md"
MARK_START, MARK_END = "<!-- BENCHMARK_TABLES:START -->", "<!-- BENCHMARK_TABLES:END -->"

STAGE_NOTES = {
    "benchmark_snaive": "Mốc chuẩn cơ sở: lặp lại doanh số cùng kỳ năm trước",
    "benchmark_ma4": "Mốc chuẩn cơ sở: trung bình 4 tuần gần nhất",
    "ml_lgb_l2": "Loss L2, tham số thô",
    "ml_xgb_l1": "Loss L1 + trọng số WMAE",
    "p1": "Hàm mất mát L1 + chặn dự báo âm",
    "p2": "Blend XGB + LGB (trọng số tối ưu WMAE) + bù đỉnh Thanksgiving α = 1.02",
    "p3_lgb": "Mô hình đơn: 8 đặc trưng thống kê nhóm + siêu tham số Optuna",
    "p3": "Optuna (TPE) + thống kê nhóm Store×Dept + Super Blend",
    "linear_ridge": "Hồi quy L2 trên thang tương đối y / quy mô chuỗi, lag + one-hot",
    "linear_lasso": "Hồi quy L1 (chọn biến) trên thang tương đối y / quy mô chuỗi",
}
METHOD_NOTES = {
    "base": "Dự báo độc lập từng tầng, vi phạm tính nhất quán",
    "bottom_up": "Giữ nguyên tầng lá, cộng dồn lên Store/Total",
    "top_down": "Phân rã Total theo tỷ trọng lịch sử",
    "ols": "Chiếu trực giao lên không gian nhất quán (W = I)",
    "wls_struct": "Trọng số theo số chuỗi lá dưới mỗi nút",
    "wls_var": "Trọng số theo phương sai phần dư từng chuỗi",
    "mint_shrink": "Hiệp phương sai phần dư co rút (Schäfer–Strimmer)",
}
COHERENT_METHODS = ["bottom_up", "top_down", "ols", "wls_struct", "wls_var", "mint_shrink"]


# =============================================================================
# ĐỊNH DẠNG
# =============================================================================

def ms(stat, fmt=",.2f"):
    """'mean ± std' cho Markdown."""
    if not stat:
        return "—"
    return f"{stat['mean']:{fmt}} ± {stat['std']:{fmt}}"


def ms_tex(stat, fmt=",.2f"):
    """'$mean \\pm std$' cho LaTeX."""
    if not stat:
        return "--"
    return f"${stat['mean']:{fmt}} \\pm {stat['std']:{fmt}}$".replace(",", "{,}")


def num_tex(value, fmt=",.2f"):
    return f"${value:{fmt}}$".replace(",", "{,}")


def tex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("≠", r"$\neq$"), ("Σ", r"$\Sigma$"), ("×", r"$\times$"), ("α", r"$\alpha$"),
                 ("→", r"$\rightarrow$"), ("–", "--"), ("σ²", r"$\sigma^2$"), ("·", r"$\cdot$"), ("≈", r"$\approx$"), ("²", r"$^2$"),
                 ("Δ", r"$\Delta$")]:
        text = text.replace(a, b)
    return text


def tex_table(caption: str, label: str, headers: list, rows: list, align: str) -> str:
    """Bảng LaTeX dạng booktabs (cần \\usepackage{booktabs})."""
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}",
             rf"\label{{{label}}}", rf"\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(rows_i) + r" \\" for rows_i in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def missing(what: str, command: str) -> str:
    return f"> ⚠️ Chưa có kết quả {what}. Chạy: `{command}`"


# =============================================================================
# BẢNG 1 — CHUỖI CẢI TIẾN ML
# =============================================================================

def _linear_stages(linear: dict) -> list:
    """Chuyển kết quả Ridge/Lasso (tất định) về cùng định dạng summary của Bảng 1 (std = 0)."""
    if not linear:
        return []
    out = []
    for name in ["Ridge", "Lasso"]:
        r = linear["results"].get(name)
        if not r:
            continue
        summary = {split: {k: {"mean": v, "std": 0.0, "n": 1} for k, v in r[split].items()} for split in ["val", "test"]}
        out.append(({"key": f"linear_{name.lower()}", "phase": "Linear Baseline",
                     "idea": f"{name} (alpha = {r['best_alpha']:g}) với lag features"}, summary))
    return out


def table1(data: dict, linear: dict = None):
    if not data:
        cmd = "python Src/Experiments/run_improvement_multiseed.py"
        return missing("chuỗi cải tiến đa seed", cmd), f"% Bảng 1: chưa có kết quả ({cmd})"
    seeds = data["seeds"]
    # Thứ tự dòng: Benchmark -> Linear Baseline -> ML Baseline -> P1 -> P2 -> P3
    rows_src = [(s, data["summary"][s["key"]]) for s in data["stages"] if s["key"].startswith("benchmark")]
    rows_src += _linear_stages(linear)
    rows_src += [(s, data["summary"][s["key"]]) for s in data["stages"] if not s["key"].startswith("benchmark")]
    md_parts, tex_parts = [], []
    for split, title in [("val", "Validation Q4/2011"), ("test", "Holdout Test 2012")]:
        best_key = min(rows_src, key=lambda x: x[1][split]["WMAE"]["mean"])[0]["key"]
        md_rows, tex_rows = [], []
        for s, summ in rows_src:
            m = summ[split]
            note = STAGE_NOTES.get(s["key"], "")
            if s["key"] in ("p2", "p3"):
                w = np.mean([r["blend_weights"][s["key"]] for r in data["runs"]], axis=0)
                note += f" (w_XGB ≈ {w[0]:.2f})"
            name = f"{s['phase']}" + (" [BEST]" if s["key"] == best_key else "")
            cells = [ms(m["WMAE"]), ms(m["MAE"]), ms(m["RMSE"]), ms(m["MASE"], ".3f"),
                     ms(m["MAPE (%)"], ".1f"), ms(m["R2"], ".4f")]
            if s["key"] == best_key:
                cells = [f"**{c}**" for c in cells[:4]] + cells[4:]
                name = f"**{name}**"
            md_rows.append([name, s["idea"], *cells, note])
            tex_cells = [ms_tex(m["WMAE"]), ms_tex(m["MAE"]), ms_tex(m["RMSE"]), ms_tex(m["MASE"], ".3f"),
                         ms_tex(m["MAPE (%)"], ".1f"), ms_tex(m["R2"], ".4f")]
            if s["key"] == best_key:
                tex_cells = [rf"\textbf{{{c}}}" for c in tex_cells[:4]] + tex_cells[4:]
            tex_rows.append([tex_escape(s["phase"] + (" [BEST]" if s["key"] == best_key else "")),
                             tex_escape(s["idea"]), *tex_cells, tex_escape(note)])
        headers = ["Phiên bản thực nghiệm", "Ý tưởng cải tiến trọng tâm", "WMAE (USD) ↓", "MAE (USD) ↓",
                   "RMSE (USD) ↓", "MASE ↓", "MAPE (%) ↓", "R² ↑", "Ghi chú & Đóng góp"]
        md_parts.append(f"**{title}** ({len(seeds)} seed: {', '.join(map(str, seeds))}; mean ± std, ddof = 1)\n\n"
                        + markdown_table(headers, md_rows,
                                         [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:", ":---"]))
        tex_parts.append(tex_table(f"Chuỗi cải tiến ML trên tập {title} (mean ± std qua {len(seeds)} seed)",
                                   f"tab:improvement_{split}", [h.replace(" ↓", "").replace(" ↑", "") for h in headers],
                                   tex_rows, "llrrrrrrl"))
    md = ("## 📊 BẢNG 1: Hướng 1 — Cải tiến Machine Learning & Ensembling (P1, P2, P3)\n\n"
          "MASE dùng mẫu số Seasonal Naive (m = 52) tính trên tập train. Benchmark là tất định (std = 0); "
          "LightGBM L2/L1 gốc không bật bagging; độ lệch nhỏ giữa các seed của chúng đến từ tính không tất định "
          "của huấn luyện đa luồng, còn XGBoost có thêm subsample 0.8. Bảng 3 dùng dự báo của seed đầu tiên (42) "
          "nên số P3 có thể khác nhẹ so với mean ở Bảng 1.\n\n" + "\n\n".join(md_parts))
    return md, "\n\n".join(tex_parts)


# =============================================================================
# BẢNG 2 & 3 — PHÂN CẤP
# =============================================================================

def merge_hierarchical(*payloads):
    """Gộp kết quả hòa giải của nhiều lần chạy (mỗi lần một nguồn tầng lá) thành một payload."""
    payloads = [p for p in payloads if p]
    if not payloads:
        return None
    merged = dict(payloads[0])
    merged["results"] = [r for p in payloads for r in p["results"]]
    return merged


def _find_source(hier: dict, name: str):
    if not hier:
        return None
    return next((r for r in hier["results"] if r["source"]["name"] == name), None)


def table2(hier: dict):
    res = _find_source(hier, "ml_baseline")
    if not res:
        cmd = "python Src/Improvement/Hierarchical/hierarchical_reconciliation.py"
        return missing("hòa giải phân cấp (nguồn ml_baseline)", cmd), f"% Bảng 2: chưa có kết quả ({cmd})"
    md_parts, tex_parts = [], []
    for split, title in [("test", "Holdout Test 2012"), ("val", "Validation Q4/2011")]:
        methods = res["methods"]
        best = min(COHERENT_METHODS, key=lambda k: methods[k][split]["metrics"]["L2_store_dept"]["WMAE"]
                   if k in methods else np.inf)
        md_rows, tex_rows = [], []
        for key, r in methods.items():
            g, m = r[split]["coherency_gap"], r[split]["metrics"]
            gap = (f"{g['store_vs_dept_pct']:.2f}% / {g['total_vs_store_pct']:.2f}%"
                   if key == "base" else "0% (nhất quán)" if g["store_vs_dept_pct"] < 1e-3 and g["total_vs_store_pct"] < 1e-3
                   else f"{g['store_vs_dept_pct']:.2e}%")
            cells = [f"{m['L0_total']['WMAE']:,.0f}", f"{m['L1_store']['WMAE']:,.0f}", f"{m['L2_store_dept']['WMAE']:,.2f}"]
            label = r["label"]
            if key == best:
                cells = [f"**{c}**" for c in cells]
                label = f"**{label}**"
            md_rows.append([label, gap, *cells, METHOD_NOTES.get(key, "")])
            tex_cells = [num_tex(m["L0_total"]["WMAE"], ",.0f"), num_tex(m["L1_store"]["WMAE"], ",.0f"),
                         num_tex(m["L2_store_dept"]["WMAE"])]
            if key == best:
                tex_cells = [rf"\textbf{{{c}}}" for c in tex_cells]
            tex_rows.append([tex_escape(r["label"]), tex_escape(gap), *tex_cells, tex_escape(METHOD_NOTES.get(key, ""))])
        headers = ["Phương pháp phân cấp", "Coherency Gap (Store≠ΣDept / Total≠ΣStore)", "WMAE L0 Total (USD)",
                   "WMAE L1 Store (USD)", "WMAE L2 Store×Dept (USD)", "Nhận xét"]
        md_parts.append(f"**{title}**\n\n" + markdown_table(headers, md_rows, [":---", ":---:", "---:", "---:", "---:", ":---"]))
        tex_parts.append(tex_table(f"Hòa giải phân cấp trên dự báo {res['source']['label']} — {title}",
                                   f"tab:hierarchical_{split}", headers, tex_rows, "lcrrrl"))
    md = (f"## 📊 BẢNG 2: Hướng 2 — Hierarchical Reconciliation (tầng lá: {res['source']['label']})\n\n"
          f"Cây {hier['hierarchy']['n_total']} Total + {hier['hierarchy']['n_store']} Store + "
          f"{hier['hierarchy']['n_leaf']:,} Store×Dept. Dự báo cơ sở Level 0/1 từ mô hình LightGBM tổng hợp riêng. "
          f"W của WLS/MinT từ phần dư `{hier.get('residual_source', 'train')}`. "
          "Dòng in đậm: phương pháp nhất quán có WMAE tầng lá thấp nhất.\n\n" + "\n\n".join(md_parts))
    return md, "\n\n".join(tex_parts)


def table3(hier: dict):
    res = _find_source(hier, "p3_best")
    if not res:
        cmd = "python Src/Improvement/Combined/run_combined.py"
        return missing("hòa giải phân cấp (nguồn p3_best)", cmd), f"% Bảng 3: chưa có kết quả ({cmd})"
    methods = res["methods"]
    order = [k for k in ["base", "bottom_up", "ols", "wls_struct", "wls_var", "mint_shrink"] if k in methods]
    coherent = [k for k in order if k != "base"]
    final = min(coherent, key=lambda k: methods[k]["val"]["metrics"]["L2_store_dept"]["WMAE"])
    md_rows, tex_rows = [], []
    for i, key in enumerate(order, 1):
        r = methods[key]
        mv, mt = r["val"]["metrics"]["L2_store_dept"], r["test"]["metrics"]["L2_store_dept"]
        g = r["val"]["coherency_gap"]
        coh = f"Lệch Store {g['store_vs_dept_pct']:.2f}%" if key == "base" else "100% nhất quán"
        name = f"{i}. P3 + {r['label']}" if key != "base" else f"{i}. P3 Best Blend (Raw)"
        if key == final:
            name += " [FINAL]"
        cells = [f"{mv['WMAE']:,.2f}", f"{mv['MAE']:,.2f}", f"{mv['MASE']:.3f}", f"{mt['WMAE']:,.2f}",
                 f"{r['test']['metrics']['L1_store']['WMAE']:,.0f}"]
        note = METHOD_NOTES.get(key, "")
        if key == final:
            md_rows.append([f"**{name}**", *[f"**{c}**" for c in cells], coh, note])
        else:
            md_rows.append([name, *cells, coh, note])
        tex_rows.append([tex_escape(name), num_tex(mv["WMAE"]), num_tex(mv["MAE"]), num_tex(mv["MASE"], ".3f"),
                         num_tex(mt["WMAE"]), num_tex(r["test"]["metrics"]["L1_store"]["WMAE"], ",.0f"),
                         tex_escape(coh), tex_escape(note)])
    headers = ["Cấu hình tích hợp", "WMAE Val L2", "MAE Val L2", "MASE Val L2", "WMAE Test L2", "WMAE Test L1 Store",
               "Coherency", "Ý nghĩa"]
    md = ("## 📊 BẢNG 3: Hướng 3 — Kết hợp P3 Best Blend + Hierarchical Reconciliation\n\n"
          "[FINAL] = cấu hình nhất quán có WMAE Validation tầng lá thấp nhất (chọn trên Val, báo cáo thêm Test).\n\n"
          + markdown_table(headers, md_rows, [":---", "---:", "---:", "---:", "---:", "---:", ":---:", ":---"]))
    tex = tex_table("P3 Best Blend kết hợp hòa giải phân cấp", "tab:p3_hierarchical", headers, tex_rows, "lrrrrrcl")
    return md, tex


# =============================================================================
# BẢNG PHỤ
# =============================================================================

def table_mint_sensitivity(hier_train: dict, hier_val: dict):
    """So sánh WLS-var / MinT khi W ước lượng từ phần dư Train (trong mẫu) và Val (ngoài mẫu) — chỉ trên Test."""
    cmd = ("python Src/Improvement/Hierarchical/hierarchical_reconciliation.py --residual-source val && "
           "python Src/Improvement/Combined/run_combined.py --residual-source val")
    if not hier_train or not hier_val:
        return missing("phân tích độ nhạy MinT (phần dư Val)", cmd), "% Bảng độ nhạy MinT: chưa có kết quả"
    md_rows, tex_rows = [], []
    for name in ["ml_baseline", "p3_best"]:
        a, b = _find_source(hier_train, name), _find_source(hier_val, name)
        if not a or not b:
            continue
        for key in ["base", "bottom_up", "wls_var", "mint_shrink"]:
            if key not in a["methods"]:
                continue
            ma, mb = a["methods"][key]["test"]["metrics"], b["methods"][key]["test"]["metrics"]
            lam_a = a["methods"][key].get("shrinkage_lambda")
            lam_b = b["methods"][key].get("shrinkage_lambda")
            lam = f"{lam_a:.3f} / {lam_b:.3f}" if lam_a is not None else "—"
            cells = [f"{ma['L0_total']['WMAE']:,.0f}", f"{ma['L1_store']['WMAE']:,.0f}", f"{ma['L2_store_dept']['WMAE']:,.2f}",
                     f"{mb['L0_total']['WMAE']:,.0f}", f"{mb['L1_store']['WMAE']:,.0f}", f"{mb['L2_store_dept']['WMAE']:,.2f}"]
            md_rows.append([a["source"]["label"], a["methods"][key]["label"], *cells, lam])
            tex_rows.append([tex_escape(a["source"]["label"]), tex_escape(a["methods"][key]["label"]),
                             *[num_tex(float(c.replace(",", ""))) for c in cells], tex_escape(lam)])
    headers = ["Tầng lá", "Phương pháp", "L0 (W: Train)", "L1 (W: Train)", "L2 (W: Train)",
               "L0 (W: Val)", "L1 (W: Val)", "L2 (W: Val)", "λ shrink (Train / Val)"]
    md = ("### Bảng phụ D: Độ nhạy của MinT theo nguồn phần dư ước lượng W (WMAE trên Holdout Test 2012)\n\n"
          "W: Train = phần dư trong mẫu 52 tuần cuối tập Train (mặc định, giống MinT gốc); "
          "W: Val = phần dư ngoài mẫu của Validation Q4/2011. Base và Bottom-Up không dùng W nên giống nhau ở hai cột; "
          "chỉ báo cáo Test vì với W từ Val, kết quả trên Val là trong mẫu.\n\n"
          + markdown_table(headers, md_rows, [":---", ":---"] + ["---:"] * 6 + [":---:"]))
    return md, tex_table("Độ nhạy của MinT theo nguồn phần dư (WMAE Test 2012)", "tab:mint_sensitivity",
                         headers, tex_rows, "ll" + "r" * 6 + "c")


def table_ablation(data: dict):
    """Ablation theo nhóm đặc trưng + Top-10 feature importance."""
    if not data:
        return missing("ablation", "python Src/Experiments/run_ablation.py"), "% Bảng ablation: chưa có kết quả"
    md_rows, tex_rows, prev = [], [], None
    for c in data["configs"]:
        s = data["summary"][c["key"]]
        delta = "—" if prev is None else f"{s['val']['WMAE']['mean'] - prev:+,.2f}"
        prev = s["val"]["WMAE"]["mean"]
        md_rows.append([c["key"], c["label"], c["n_features"], ms(s["val"]["WMAE"]), delta, ms(s["val"]["MASE"], ".3f"),
                        ms(s["test"]["WMAE"]), ms(s["test"]["MASE"], ".3f")])
        tex_rows.append([c["key"], tex_escape(c["label"]), str(c["n_features"]), ms_tex(s["val"]["WMAE"]),
                         tex_escape(delta), ms_tex(s["val"]["MASE"], ".3f"), ms_tex(s["test"]["WMAE"]),
                         ms_tex(s["test"]["MASE"], ".3f")])
    headers = ["Cấu hình", "Nhóm đặc trưng (cộng dồn)", "#Đặc trưng", "Val WMAE", "Δ Val WMAE", "Val MASE",
               "Test WMAE", "Test MASE"]
    imp = data.get("feature_importance_top20", [])[:10]
    imp_md = markdown_table(["#", "Đặc trưng", "Gain (%)", "Split count (%)"],
                            [[i + 1, r["feature"], f"{r['gain_pct']:.2f}%", f"{r['split_pct']:.2f}%"]
                             for i, r in enumerate(imp)], ["---:", ":---", "---:", "---:"])
    md = (f"### Bảng phụ E: Ablation study theo nhóm đặc trưng ({data['model']}, {len(data['seeds'])} seed)\n\n"
          "Δ = thay đổi Val WMAE so với cấu hình liền trước (âm = tốt hơn).\n\n"
          + markdown_table(headers, md_rows, [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"])
          + "\n\n**Top-10 feature importance** (cấu hình C5, theo total gain; hình: "
          "`results/06_ablation/feature_importance.png`)\n\n" + imp_md)
    tex = tex_table("Ablation study theo nhóm đặc trưng (mean ± std qua các seed)", "tab:ablation",
                    headers, tex_rows, "llrrrrrr")
    return md, tex


def table_cv(suffix_quick: bool):
    rows, tex_rows = [], []
    for path in sorted(CV_DIR.glob("cv_*.json")):
        r = load_json(path)
        if not r or bool(r.get("quick")) != suffix_quick:
            continue
        s = r["summary"]
        fold_wmae = " / ".join(f"{f['metrics']['WMAE']:,.0f}" for f in r["folds"])
        rows.append([r["model"], r["feature_mode"], len(r["folds"]), ms(s["MAE"]), ms(s["RMSE"]), ms(s["MASE"], ".3f"),
                     ms(s["WMAE"]), fold_wmae])
        tex_rows.append([r["model"], tex_escape(r["feature_mode"]), str(len(r["folds"])), ms_tex(s["MAE"]),
                         ms_tex(s["RMSE"]), ms_tex(s["MASE"], ".3f"), ms_tex(s["WMAE"]), tex_escape(fold_wmae)])
    if not rows:
        return missing("Rolling Origin CV", "python Src/Evaluation/time_series_cv.py"), "% Bảng CV: chưa có kết quả"
    headers = ["Mô hình", "Feature mode", "#Fold", "MAE", "RMSE", "MASE", "WMAE", "WMAE từng fold"]
    md = ("### Bảng phụ A: Rolling Origin Cross-Validation (expanding window, 4 fold × 13 tuần; mean ± std qua fold)\n\n"
          + markdown_table(headers, rows, [":---", ":---:", ":---:", "---:", "---:", "---:", "---:", ":---"]))
    return md, tex_table("Rolling Origin CV (mean ± std qua 4 fold)", "tab:cv", headers, tex_rows, "lccrrrrl")


def table_multiseed(data: dict):
    if not data:
        return missing("multi-seed", "python Src/Experiments/run_multiseed.py"), "% Bảng multi-seed: chưa có kết quả"
    rows, tex_rows = [], []
    for r in data["results"]:
        s = r["summary"]
        rows.append([r["model"], r["device"], r["params_source"], len(r["seeds"]), ms(s["val"]["WMAE"]),
                     ms(s["test"]["WMAE"]), ms(s["test"]["MAE"]), ms(s["test"]["RMSE"]), ms(s["test"]["MASE"], ".3f"),
                     ms(s["cost"]["fit_time_s"], ",.1f"), f"{s['cost']['inference_time_s']['mean']:.2f}",
                     ms(s["cost"]["n_parameters"], ",.0f")])
        tex_rows.append([r["model"], r["device"], tex_escape(r["params_source"]), str(len(r["seeds"])),
                         ms_tex(s["val"]["WMAE"]), ms_tex(s["test"]["WMAE"]), ms_tex(s["test"]["MAE"]),
                         ms_tex(s["test"]["RMSE"]), ms_tex(s["test"]["MASE"], ".3f"),
                         ms_tex(s["cost"]["fit_time_s"], ",.1f"), num_tex(s["cost"]["inference_time_s"]["mean"]),
                         ms_tex(s["cost"]["n_parameters"], ",.0f")])
    headers = ["Mô hình", "Thiết bị", "Tham số", "#Seed", "Val WMAE", "Test WMAE", "Test MAE", "Test RMSE",
               "Test MASE", "Fit time (s)", "Inference (s)", "Số tham số"]
    md = ("### Bảng phụ B: 3 mô hình GBDT — đa seed, thời gian huấn luyện & số tham số\n\n"
          "Số tham số = tổng số lá + tổng số node split của toàn bộ cây.\n\n"
          + markdown_table(headers, rows, [":---", ":---:", ":---:", ":---:"] + ["---:"] * 8))
    return md, tex_table("Ba mô hình GBDT: mean ± std qua các seed, chi phí tính toán", "tab:multiseed",
                         headers, tex_rows, "lccc" + "r" * 8)


def table_tuning(data: dict):
    if not data:
        return missing("tuning", "python Src/Experiments/tune.py"), "% Bảng tuning: chưa có kết quả"
    rows, tex_rows = [], []
    for r in data["results"]:
        params = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["params"].items())
        rows.append([r["model"], f"{r['best_val_wmae']:,.2f}", r["n_trials_completed"],
                     f"{r['total_time_s'] / 60:,.1f}", params])
        tex_rows.append([r["model"], num_tex(r["best_val_wmae"]), str(r["n_trials_completed"]),
                         num_tex(r["total_time_s"] / 60, ",.1f"), tex_escape(params)])
    b = data["results"][0]["budget"]
    headers = ["Mô hình", "Best Val WMAE", "Số trial", "Thời gian (phút)", "Siêu tham số tốt nhất"]
    md = (f"### Bảng phụ C: Tuning Optuna với ngân sách đồng nhất ({b['n_trials']} trials / {b['timeout_minutes']} phút)\n\n"
          + markdown_table(headers, rows, [":---", "---:", "---:", "---:", ":---"]))
    return md, tex_table("Kết quả tuning với ngân sách đồng nhất", "tab:tuning", headers, tex_rows, "lrrrl")


def hardware_section(hw: dict) -> str:
    if not hw:
        return ""
    gpus = "; ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in hw.get("gpus", [])) or "Không có"
    return ("### Cấu hình phần cứng\n\n"
            + markdown_table(["Thành phần", "Chi tiết"], [
                ["CPU", f"{hw['cpu']['name']} ({hw['cpu']['physical_cores']} nhân / {hw['cpu']['logical_threads']} luồng)"],
                ["RAM", f"{hw['ram_gb']} GB"], ["GPU", gpus], ["Hệ điều hành", hw["os"]], ["Python", hw["python"]],
                ["Thư viện", ", ".join(f"{k} {v}" for k, v in hw["libraries"].items())],
            ], [":---", ":---"]))


# =============================================================================
# MAIN
# =============================================================================

def update_docs(md: str):
    """
    Chèn/ghi đè khối bảng giữa hai marker trong Final_Analyze.md. Khối nằm dưới mục 7.x của tài liệu nên
    mọi heading được hạ 2 cấp (## -> ####) và bỏ tiêu đề cấp 1 của file benchmark độc lập.
    """
    text = DOC_PATH.read_text(encoding="utf-8")
    lines = md.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(("##" + l) if l.startswith("#") else l for l in lines).strip()
    block = f"{MARK_START}\n{body}\n{MARK_END}"
    if MARK_START in text and MARK_END in text:
        head, rest = text.split(MARK_START, 1)
        _, tail = rest.split(MARK_END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    DOC_PATH.write_text(text, encoding="utf-8")
    print(f"[Benchmark] Đã cập nhật bảng vào {DOC_PATH.name}")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Sinh 3 bảng đối sánh trọng tâm (Markdown + LaTeX)")
    parser.add_argument("--quick", action="store_true", help="Đọc kết quả chạy thử *_quick")
    parser.add_argument("--update-docs", action="store_true", help="Chèn bảng Markdown vào Final_Analyze.md")
    args = parser.parse_args()
    sfx = "_quick" if args.quick else ""

    improvement = load_json(get_result_dir("improvement_multiseed") / f"improvement_multiseed{sfx}.json")
    linear = load_json(get_result_dir("baseline") / f"linear_results{sfx}.json")
    ablation = load_json(get_result_dir("ablation") / f"ablation_results{sfx}.json")
    # Hướng 2 (nguồn ml_baseline) và Hướng 3 (nguồn p3_best) lưu ở hai thư mục riêng -> gộp để dựng bảng
    hier = merge_hierarchical(load_json(HIER_DIR / f"hierarchical_results{sfx}.json"),
                              load_json(COMB_DIR / f"combined_results{sfx}.json"))
    hier_resval = merge_hierarchical(load_json(HIER_DIR / f"hierarchical_results{sfx}_resval.json"),
                                     load_json(COMB_DIR / f"combined_results{sfx}_resval.json"))
    multiseed = load_json(get_result_dir("multiseed") / f"multiseed_summary{sfx}.json")
    tuning = load_json(get_result_dir("tuning") / f"tuning_comparison{sfx}.json")
    hardware = load_json(HARDWARE_INFO_PATH) or (multiseed or {}).get("hardware")

    # Bảng chính 1-3, sau đó bảng phụ A-E theo đúng thứ tự chữ cái
    sections = [table1(improvement, linear), table2(hier), table3(hier), table_cv(args.quick),
                table_multiseed(multiseed), table_tuning(tuning), table_mint_sensitivity(hier, hier_resval),
                table_ablation(ablation)]
    title = "# 📊 BẢNG ĐỐI SÁNH TRỌNG TÂM (tự động sinh bởi `Src/Evaluation/generate_benchmark_tables.py`)"
    if args.quick:
        title += "\n\n> ⚠️ Đây là kết quả CHẠY THỬ (`--quick`, 10% chuỗi, ít cây) — không dùng cho báo cáo."
    md = title + "\n\n" + "\n\n---\n\n".join(s[0] for s in sections)
    hw_md = hardware_section(hardware)
    if hw_md:
        md += "\n\n---\n\n" + hw_md
    tex = ("% Tự động sinh bởi Src/Evaluation/generate_benchmark_tables.py — cần \\usepackage{booktabs,graphicx}\n\n"
           + "\n\n".join(s[1] for s in sections))

    md_path, tex_path = RESULT_DIR / f"benchmark_tables{sfx}.md", RESULT_DIR / f"benchmark_tables{sfx}.tex"
    md_path.write_text(md + "\n", encoding="utf-8")
    tex_path.write_text(tex + "\n", encoding="utf-8")
    print(md)
    print(f"\n[Benchmark] Đã lưu: {md_path}\n[Benchmark] Đã lưu: {tex_path}")
    if args.update_docs:
        if args.quick:
            print("[Benchmark] Bỏ qua --update-docs ở chế độ --quick.")
        else:
            update_docs(md)


if __name__ == "__main__":
    main()
