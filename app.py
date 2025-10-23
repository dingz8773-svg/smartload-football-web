# -*- coding: utf-8 -*-
import io, os, json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from zipfile import ZipFile
from io import BytesIO

# --- Page config ---
st.set_page_config(page_title="SmartLoad-Football · Dashboard (EN + 中文建议)", layout="wide")
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# --- Title ---
st.title("SmartLoad-Football · Dashboard (EN)")
st.caption("Team heatmap • Individual trends • Yellow/Red alert list • Per-player advice (中文)")

# --- Sidebar ---
with st.sidebar:
    st.header("Data & Settings")
    file = st.file_uploader("Upload CSV or JSON (array)", type=["csv", "json"])
    st.markdown("---")
    st.subheader("Thresholds")
    acwr_th = st.number_input("ACWR risk threshold", value=0.856, step=0.05, format="%.2f")
    y_th = st.number_input("FRI yellow threshold (≥)", value=0.241, step=0.05, format="%.2f")
    r_th = st.number_input("FRI red threshold (≥)", value=0.311, step=0.05, format="%.2f")
    recompute = st.checkbox("Recompute band from FRI thresholds (ignore original band)", value=False)
    st.caption("Tip: Enable if your data has no band column or uses different thresholds.")

# --- Data helpers ---
def read_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        data = uploaded_file.read()
        for enc in ("utf-8-sig", "gbk"):  # handle Excel CSV too
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                continue
        raise ValueError("CSV encoding not recognized (tried UTF-8-SIG and GBK).")
    else:
        data = json.load(uploaded_file)
        if isinstance(data, dict):
            data = data.get("data", [])
        return pd.DataFrame(data)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # 仅强制 player_id / date；FRI 等字段若缺失或无差异，进入“演示模式”自动生成/扰动
    required = ["player_id", "date"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 转数值（若存在）
    for c in ["FRI", "ACWR_7_28", "HRV_ratio", "CMJ_change_pct", "sRPE_7d", "minutes_last_game"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------- 演示模式（制造差异） ----------------
    rng = np.random.default_rng(42)

    # FRI：如果不存在或全部缺失，则生成；如果几乎没有差异，加入小抖动
    if "FRI" not in df.columns or df["FRI"].isna().all():
        df["FRI"] = rng.uniform(0.2, 0.9, len(df))
    else:
        if df["FRI"].nunique(dropna=True) <= 2 or (df["FRI"].std(skipna=True) or 0) < 0.02:
            jitter = pd.Series(rng.normal(0, 0.03, len(df)), index=df.index)
            df["FRI"] = (df["FRI"].fillna(df["FRI"].median()) + jitter).clip(0, 1)

    # ACWR：若缺失或无差异，按 0.8–1.6 生成或抖动
    if "ACWR_7_28" not in df.columns or df["ACWR_7_28"].isna().all():
        df["ACWR_7_28"] = rng.uniform(0.8, 1.6, len(df))
    else:
        if df["ACWR_7_28"].nunique(dropna=True) <= 2 or (df["ACWR_7_28"].std(skipna=True) or 0) < 0.02:
            df["ACWR_7_28"] = (df["ACWR_7_28"].fillna(1.0) + rng.normal(0, 0.05, len(df))).clip(0.6, 2.0)

    # HRV 比例：缺失则 0.85–1.10；无差异微抖动
    if "HRV_ratio" not in df.columns or df["HRV_ratio"].isna().all():
        df["HRV_ratio"] = rng.uniform(0.85, 1.10, len(df))
    else:
        if df["HRV_ratio"].nunique(dropna=True) <= 2 or (df["HRV_ratio"].std(skipna=True) or 0) < 0.01:
            df["HRV_ratio"] = (df["HRV_ratio"].fillna(1.0) + rng.normal(0, 0.01, len(df))).clip(0.80, 1.20)

    # CMJ 变化：缺失则 -10% ~ +5%；无差异微抖动
    if "CMJ_change_pct" not in df.columns or df["CMJ_change_pct"].isna().all():
        df["CMJ_change_pct"] = rng.uniform(-10, 5, len(df))
    else:
        if df["CMJ_change_pct"].nunique(dropna=True) <= 2 or (df["CMJ_change_pct"].std(skipna=True) or 0) < 0.5:
            df["CMJ_change_pct"] = df["CMJ_change_pct"].fillna(0) + rng.normal(0, 0.8, len(df))

    # band
    if "band" in df.columns:
        df["band"] = df["band"].astype(str).str.lower()
    else:
        df["band"] = ""

    return df
    # ---------------- 演示模式结束 ----------------

if not file:
    st.info("Upload a multi-day CSV/JSON to start.")
    st.stop()

try:
    df = read_df(file)
    df = normalize(df)
except Exception as e:
    st.error(f"Failed to read/parse data: {e}")
    st.stop()

# Optional: recompute band from thresholds
if recompute:
    def band_from_fri(x: float) -> str:
        if pd.isna(x): return "green"
        if x >= r_th: return "red"
        if x >= y_th: return "yellow"
        return "green"
    df["band"] = df["FRI"].apply(band_from_fri)

# ------------------ 中文建议引擎（规则驱动） ------------------
def build_reasons(ctx):
    """生成触发原因（中文）"""
    reasons = []
    if ctx.get("FRI") is not None:
        if ctx["FRI"] >= r_th:
            reasons.append(f"FRI {ctx['FRI']:.2f} ≥ 红灯阈值 {r_th:.2f}")
        elif ctx["FRI"] >= y_th:
            reasons.append(f"FRI {ctx['FRI']:.2f} ≥ 黄灯阈值 {y_th:.2f}")
    acwr = ctx.get("ACWR_7_28")
    if acwr is not None:
        if acwr >= acwr_th:
            reasons.append(f"ACWR {acwr:.2f} ≥ 风险阈值 {acwr_th:.2f}")
        elif acwr >= 1.30:
            reasons.append(f"ACWR {acwr:.2f} 有上升趋势")
    hrv = ctx.get("HRV_ratio")
    if hrv is not None and hrv < 0.95:
        reasons.append(f"HRV 比例 {hrv:.2f} 低于基线")
    cmj = ctx.get("CMJ_change_pct")
    if cmj is not None and cmj <= -5:
        reasons.append(f"CMJ 较基线下降 {cmj:.1f}%")
    return reasons

def build_plan(ctx):
    """根据 band 与指标生成训练/恢复/监测建议（中文）"""
    band = ctx.get("band", "green")
    acwr = ctx.get("ACWR_7_28")
    hrv = ctx.get("HRV_ratio")
    cmj = ctx.get("CMJ_change_pct")

    plan = {"training": [], "recovery": [], "monitoring": []}

    if band == "red":
        plan["training"].append("以技战术为主；高速跑/对抗 ≤ 正常的 60%")
        if acwr is not None and acwr >= acwr_th:
            plan["training"].append("今日总量下调 30–40%，取消一段高强度模块")
        if cmj is not None and cmj <= -5:
            plan["training"].append("暂停弹跳与大幅度变向训练")
        plan["recovery"].append("优先睡眠；呼吸训练 + 冷热交替；建议理疗评估")
        if hrv is not None and hrv <= 0.90:
            plan["recovery"].append("20–30 分钟低强度有氧 + 10 分钟呼吸放松")
        plan["monitoring"].extend(["24 小时内复测 HRV/CMJ", "若持续红灯，上报教练与医务"])
    elif band == "yellow":
        plan["training"].append("保留主训练；高速或对抗减少 1 组")
        if acwr is not None and acwr >= 1.30:
            plan["training"].append("今日总量下调 15–20%")
        if cmj is not None and cmj <= -5:
            plan["training"].append("降低离心负荷，限制跳跃次数")
        if hrv is not None and hrv < 0.95:
            plan["recovery"].append("20–30 分钟低强度有氧 + 10 分钟呼吸放松")
        plan["monitoring"].append("48 小时内复测 HRV/CMJ")
    else:  # green
        plan["training"].append("维持当前计划；可加 1–2 组短距离高质量冲刺")
        plan["recovery"].append("常规拉伸与补水；睡眠 ≥ 8 小时")
        plan["monitoring"].append("晨间 HRV + 简短 CMJ 例行监测")

    # 去重
    for k in plan:
        seen, out = set(), []
        for s in plan[k]:
            if s and s not in seen:
                out.append(s); seen.add(s)
        plan[k] = out
    return plan

def render_advice_text(ctx):
    """拼装中文建议文本"""
    reasons = build_reasons(ctx)
    plan = build_plan(ctx)
    band = ctx.get("band", "green")
    head = (
        f"今日 FRI {ctx.get('FRI', float('nan')):.2f}（{band}）— "
        + ("；".join(reasons) if reasons else "未见显著异常")
        + "。"
    )
    text = (
        head + "\n"
        + "训练建议：" + ("；".join(plan["training"]) if plan["training"] else "维持当前计划") + "。\n"
        + "恢复建议：" + ("；".join(plan["recovery"]) if plan["recovery"] else "常规拉伸、补水与高质量睡眠") + "。\n"
        + "监测建议：" + ("；".join(plan["monitoring"]) if plan["monitoring"] else "次日复测 HRV/CMJ") + "。\n"
    )
    return text, reasons, plan

# ============= Layout =============
left, right = st.columns([1.1, 1.3], gap="large")

# ===== Left: Team FRI Heatmap + 个体建议 =====
with left:
    st.subheader("Team FRI Heatmap")
    fri = df.pivot_table(index="player_id",
                         columns=df["date"].dt.date,
                         values="FRI", aggfunc="mean").sort_index()
    if not fri.empty:
        fig, ax = plt.subplots()
        im = ax.imshow(fri.values, aspect="auto")
        ax.set_yticks(range(len(fri.index))); ax.set_yticklabels(fri.index)
        ax.set_xticks(range(len(fri.columns)))
        ax.set_xticklabels([d.strftime("%m-%d") for d in fri.columns], rotation=45, ha="right")
        ax.set_title("SmartLoad-Football · Team FRI Heatmap")
        fig.colorbar(im, ax=ax).ax.set_ylabel("FRI", rotation=270, labelpad=15)
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No FRI data to visualize.")

    # 标题去掉“（热力图下方）”
    st.markdown("### 个体指导建议")
    all_dates = sorted(df["date"].dt.date.dropna().unique().tolist())
    pid_adv = st.selectbox("选择球员（建议）", sorted(df["player_id"].unique()), index=0, key="adv_pid")
    date_adv = st.selectbox("选择日期（建议）", all_dates, index=len(all_dates)-1,
                            format_func=lambda d: d.strftime("%Y-%m-%d"), key="adv_date")

    sub_adv = df[(df["player_id"] == pid_adv) & (df["date"].dt.date == date_adv)]
    if sub_adv.empty:
        st.info("该球员在该日期没有数据。请更换球员或日期。")
    else:
        row = sub_adv.iloc[0].to_dict()
        ctx = {
            "FRI": row.get("FRI"),
            "band": row.get("band", "green"),
            "ACWR_7_28": row.get("ACWR_7_28"),
            "HRV_ratio": row.get("HRV_ratio"),
            "CMJ_change_pct": row.get("CMJ_change_pct"),
            "sRPE_7d": row.get("sRPE_7d"),
            "minutes_last_game": row.get("minutes_last_game"),
        }
        advice_text, _, _ = render_advice_text(ctx)
        st.code(advice_text, language="text")

# ===== Right: Individual trends（保持不变） =====
def _xtick_format(ax, dates):
    n = len(dates)
    if n <= 8:
        idx = list(range(n))
    else:
        step = max(1, n // 8)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1: idx.append(n - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([dates[i].strftime("%Y-%m-%d") for i in idx], rotation=45, ha="right")

with right:
    st.subheader("Individual Trends (lines + markers + thresholds)")
    pids = sorted(df["player_id"].unique())
    pid = st.selectbox("Select player (charts)", pids, index=0)
    sub = df[df["player_id"] == pid].sort_values("date").reset_index(drop=True)
    dates = sub["date"].dt.date.tolist()

    # Chart 1: ACWR + threshold
    fig1, ax1 = plt.subplots()
    if "ACWR_7_28" in sub.columns and sub["ACWR_7_28"].notna().any():
        ax1.plot(range(len(sub)), sub["ACWR_7_28"], marker="o")
        ax1.axhline(acwr_th, linestyle="--", linewidth=2, color="red")
        ax1.legend(["ACWR", f"Risk threshold {acwr_th:.2f}"], loc="best")
    else:
        ax1.plot(range(len(sub)), sub["FRI"], marker="o")
        ax1.legend(["FRI (ACWR not provided)"], loc="best")
    _xtick_format(ax1, dates)
    ax1.set_title(f"{pid} · ACWR 14-day Trend (example)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("ACWR")
    st.pyplot(fig1, use_container_width=True)

    # Chart 2: FRI + yellow/red thresholds
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(sub)), sub["FRI"], marker="o")
    ax2.axhline(y_th, linestyle="--", linewidth=2, color="green")
    ax2.axhline(r_th, linestyle="--", linewidth=2, color="red")
    ax2.legend(["FRI", f"Yellow {y_th:.2f}", f"Red {r_th:.2f}"], loc="best")
    _xtick_format(ax2, dates)
    ax2.set_title(f"{pid} · FRI 14-day Trend (example)")
    ax2.set_xlabel("Date"); ax2.set_ylabel("FRI index")
    st.pyplot(fig2, use_container_width=True)

# ===== Bottom: alert list (select date + download + export advice) =====
st.subheader("Yellow/Red Alert List (by date)")
all_dates_tbl = sorted(df["date"].dt.date.dropna().unique().tolist())
sel_date = st.selectbox("Select date (table)", all_dates_tbl, index=len(all_dates_tbl)-1,
                        format_func=lambda d: d.strftime("%Y-%m-%d"))
day_df = df[df["date"].dt.date == sel_date].copy()

def advice_for_row(row):
    ctx = {
        "FRI": row.get("FRI"),
        "band": row.get("band", "green"),
        "ACWR_7_28": row.get("ACWR_7_28"),
        "HRV_ratio": row.get("HRV_ratio"),
        "CMJ_change_pct": row.get("CMJ_change_pct"),
        "sRPE_7d": row.get("sRPE_7d"),
        "minutes_last_game": row.get("minutes_last_game"),
    }
    text, reasons, plan = render_advice_text(ctx)
    return text, " | ".join(reasons), " | ".join(plan["training"]), " | ".join(plan["recovery"]), " | ".join(plan["monitoring"])

texts, reasons_col, train_col, recov_col, monitor_col = [], [], [], [], []
for _, r in day_df.iterrows():
    t, rs, tr, rc, mo = advice_for_row(r)
    texts.append(t); reasons_col.append(rs); train_col.append(tr); recov_col.append(rc); monitor_col.append(mo)

day_df["reasons"] = reasons_col
day_df["training"] = train_col
day_df["recovery"] = recov_col
day_df["monitoring"] = monitor_col
day_df["advice_text"] = texts

alerts = day_df[day_df["band"].isin(["yellow", "red"])][
    ["player_id","date","FRI","band","ACWR_7_28","HRV_ratio","CMJ_change_pct",
     "reasons","training","recovery","monitoring","advice_text"]
].sort_values(["band","FRI"], ascending=[True, False])

if alerts.empty:
    st.info("No yellow/red alerts for this date. Try other dates or adjust thresholds.")
else:
    st.dataframe(alerts.drop(columns=["advice_text"]), use_container_width=True, height=260)
    csv_bytes = alerts.drop(columns=["advice_text"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("下载当日预警清单（CSV）", data=csv_bytes,
                       file_name=f"alerts_{sel_date}.csv", mime="text/csv")
    mem = BytesIO()
    with ZipFile(mem, "w") as zf:
        for _, r in alerts.iterrows():
            pid = str(r["player_id"])
            zf.writestr(f"{pid}.txt", r["advice_text"])
    st.download_button("下载个人建议合集（ZIP）", data=mem.getvalue(),
                       file_name=f"player_advice_{sel_date}.zip", mime="application/zip")
