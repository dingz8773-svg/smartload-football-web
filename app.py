# -*- coding: utf-8 -*-
import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -----------------------------------------------------------------------------
# 页面配置
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SmartLoad-Football · 管理后台 (Final Demo v4)", layout="wide")

# -----------------------------------------------------------------------------
# 中文字体注册（关键改动）
# - 随项目携带 fonts/NotoSansSC-Regular.otf 或 fonts/SimHei.ttf
# - 找到就 addfont，并把 rcParams 的中文无衬线优先级放在最前
# - 找不到则回退到系统常见中文字体与 DejaVu Sans
# -----------------------------------------------------------------------------
FONT_CANDIDATES = [
    "fonts/NotoSansSC-Regular.otf",  # 推荐：思源黑体
    "fonts/SimHei.ttf",              # 备选：黑体
    "fonts/MicrosoftYaHei.ttf",      # 若你项目里放了也会被加载
]
loaded_font_names = []  # 记录成功加入的字体家族名

for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
            # 取出字体家族名（matplotlib 会识别文件内的 family）
            try:
                prop = fm.FontProperties(fname=fp)
                family = prop.get_name()
            except Exception:
                # 若无法解析家族名，就按常见名字猜测
                family = "Noto Sans SC" if "NotoSansSC" in fp else (
                         "SimHei" if "SimHei" in fp else "Microsoft YaHei")
            loaded_font_names.append(family)
        except Exception:
            pass

# 将本地字体（若有）放在优先级最前，后面是系统常见中文字体 & 兜底
matplotlib.rcParams["font.sans-serif"] = loaded_font_names + [
    "Noto Sans SC", "SimHei", "Microsoft YaHei", "Source Han Sans SC",
    "Heiti SC", "Arial Unicode MS", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# -----------------------------------------------------------------------------
# 标题
# -----------------------------------------------------------------------------
st.title("SmartLoad-Football · 管理后台 (Final Demo v4)")
st.caption("团队热力图、个体趋势（折线+阈值虚线+圆点）、黄/红灯预警清单（中文图例 & 更清晰坐标）")

# -----------------------------------------------------------------------------
# 侧栏：数据上传与阈值
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("数据与设置")
    file = st.file_uploader("上传 CSV / JSON（数组）", type=["csv", "json"])
    st.markdown("---")
    st.subheader("阈值设置")
    acwr_th = st.number_input("ACWR 风险阈值", value=1.50, step=0.05, format="%.2f")
    y_th = st.number_input("FRI · 黄灯阈值（≥）", value=0.50, step=0.05, format="%.2f")
    r_th = st.number_input("FRI · 红灯阈值（≥）", value=0.70, step=0.05, format="%.2f")
    recompute = st.checkbox("按阈值从 FRI 重算 band（忽略原 band）", value=False)
    st.caption("提示：若原数据无 band 或阈值不同，可勾选此项让清单按上方阈值重算。")

# -----------------------------------------------------------------------------
# 读数据
# -----------------------------------------------------------------------------
def read_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        data = uploaded_file.read()
        # 先尝试 UTF-8-SIG，失败再回退 GBK（Excel 常见）
        for enc in ("utf-8-sig", "gbk"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                continue
        raise ValueError("CSV 编码无法识别（已尝试 UTF-8-SIG 与 GBK）")
    else:
        data = json.load(uploaded_file)
        if isinstance(data, dict):
            data = data.get("data", [])
        return pd.DataFrame(data)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    req = ["player_id", "date", "FRI"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"缺少必填列：{c}")
    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["FRI", "ACWR_7_28", "HRV_ratio", "CMJ_change_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "band" in df.columns:
        df["band"] = df["band"].astype(str).str.lower()
    else:
        df["band"] = ""
    return df

if not file:
    st.info("请先上传包含多天数据的 CSV/JSON。")
    st.stop()

try:
    df = read_df(file)
    df = normalize(df)
except Exception as e:
    st.error(f"读取/解析失败：{e}")
    st.stop()

# 可选：按阈值重算 band
if recompute:
    def band_from_fri(x: float) -> str:
        if pd.isna(x):
            return "green"
        if x >= r_th:
            return "red"
        if x >= y_th:
            return "yellow"
        return "green"
    df["band"] = df["FRI"].apply(band_from_fri)

# -----------------------------------------------------------------------------
# 左侧：团队热力图
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1.3], gap="large")
with left:
    st.subheader("团队 FRI 热力图")
    fri = df.pivot_table(index="player_id",
                         columns=df["date"].dt.date,
                         values="FRI",
                         aggfunc="mean").sort_index()
    if not fri.empty:
        fig, ax = plt.subplots()
        im = ax.imshow(fri.values, aspect="auto")
        ax.set_yticks(range(len(fri.index)))
        ax.set_yticklabels(fri.index)
        ax.set_xticks(range(len(fri.columns)))
        ax.set_xticklabels([d.strftime("%m-%d") for d in fri.columns],
                           rotation=45, ha="right")
        ax.set_title("SmartLoad-Football · Team FRI Heatmap")
        fig.colorbar(im, ax=ax).ax.set_ylabel("FRI", rotation=270, labelpad=15)
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("没有可视化的数据。")

# -----------------------------------------------------------------------------
# 右侧：个体趋势（折线 + 圆点 + 阈值虚线）
# -----------------------------------------------------------------------------
def _xtick_format(ax, dates):
    """将横轴日期抽稀并旋转，最多显示约 8 个刻度。"""
    n = len(dates)
    if n <= 8:
        idx = list(range(n))
    else:
        step = max(1, n // 8)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([dates[i].strftime("%Y-%m-%d") for i in idx],
                       rotation=45, ha="right")

with right:
    st.subheader("个体趋势（折线+阈值虚线+圆点）")
    pids = sorted(df["player_id"].unique())
    pid = st.selectbox("选择球员", pids, index=0)
    sub = df[df["player_id"] == pid].sort_values("date").reset_index(drop=True)
    dates = sub["date"].dt.date.tolist()

    # 图1：ACWR & 阈值
    fig1, ax1 = plt.subplots()
    if "ACWR_7_28" in sub.columns and sub["ACWR_7_28"].notna().any():
        ax1.plot(range(len(sub)), sub["ACWR_7_28"], marker="o")
        ax1.axhline(acwr_th, linestyle="--", linewidth=2, color="red")
        ax1.legend(["ACWR", f"风险阈值 {acwr_th:.2f}"], loc="best")
    else:
        ax1.plot(range(len(sub)), sub["FRI"], marker="o")
        ax1.legend(["FRI（无 ACWR 列，临时展示）"], loc="best")
    _xtick_format(ax1, dates)
    ax1.set_title(f"{pid} · ACWR 14天趋势（示意）")
    ax1.set_xlabel("日期"); ax1.set_ylabel("ACWR 值")
    st.pyplot(fig1, use_container_width=True)

    # 图2：FRI & 黄/红阈值
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(sub)), sub["FRI"], marker="o")
    ax2.axhline(y_th, linestyle="--", linewidth=2, color="green")
    ax2.axhline(r_th, linestyle="--", linewidth=2, color="red")
    ax2.legend(["FRI", f"黄灯阈值 {y_th:.2f}", f"红灯阈值 {r_th:.2f}"], loc="best")
    _xtick_format(ax2, dates)
    ax2.set_title(f"{pid} · 疲劳风险指数（FRI）14天趋势（示意）")
    ax2.set_xlabel("日期"); ax2.set_ylabel("FRI 指数")
    st.pyplot(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# 底部：预警清单（可选日期 + 下载）
# -----------------------------------------------------------------------------
st.subheader("当日黄/红灯预警清单")
all_dates = sorted(df["date"].dt.date.dropna().unique().tolist())
sel_date = st.selectbox("选择日期", all_dates,
                        index=len(all_dates) - 1,
                        format_func=lambda d: d.strftime("%Y-%m-%d"))
day_df = df[df["date"].dt.date == sel_date]
alerts = day_df[day_df["band"].isin(["yellow", "red"])][
    ["player_id", "date", "FRI", "band", "ACWR_7_28", "HRV_ratio", "CMJ_change_pct"]
].sort_values(["band", "FRI"], ascending=[True, False])

if alerts.empty:
    st.info("该日期没有黄/红灯。可调整上方阈值或选择其它日期。")
else:
    st.dataframe(alerts, use_container_width=True, height=240)
    csv_bytes = alerts.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载 CSV", data=csv_bytes,
                       file_name=f"alerts_{sel_date}.csv",
                       mime="text/csv")
