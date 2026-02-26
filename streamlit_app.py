import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import warnings

warnings.filterwarnings("ignore")

# =========================
# 0. 基本配置
# =========================
MODEL_PATH = "svm_model.pkl"
BACKGROUND_CSV = "testdata.csv"

FEATURES = [
    "Age",
    "StoolFrequencyPerDay",
    "Platelets",
    "MCH",
    "WBC",
    "HDL_Cholesterol",
]

# UC=1, CD=0
UC_LABEL = 1
CD_LABEL = 0

# 阈值提示（可按需调整）
UC_HIGH_TH = 0.60
UC_BORDER_LOW = 0.40  # 0.40~0.60 视为边界/不确定区

# =========================
# 1. 缓存加载
# =========================
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

@st.cache_data
def load_background(csv_path: str):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"背景数据缺少列：{missing}")
    df = df[FEATURES].dropna()
    if df.shape[0] < 5:
        raise ValueError("背景数据行数过少（<5），LIME 解释可能不稳定。请提供更多样本。")
    return df

model = load_model(MODEL_PATH)
X_bg = load_background(BACKGROUND_CSV)

# =========================
# 2. 页面
# =========================
st.set_page_config(page_title="UC vs CD 预测器", layout="wide")
st.title("UC vs CD 预测器（UC=1，CD=0）")
st.caption("支持单样本输入 + 批量CSV预测 + LIME单样本解释（局部可解释）")

tab1, tab2 = st.tabs(["🧍 单样本预测", "📁 批量CSV预测"])

# =========================
# 3. 工具函数：概率映射（确保 UC=1/CD=0 对应正确概率）
# =========================
def get_classes_list(m):
    if hasattr(m, "classes_"):
        return list(m.classes_)
    if hasattr(m, "named_steps"):
        last = list(m.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return list(last.classes_)
    return None

CLASSES = get_classes_list(model)

def proba_of_label(proba_row: np.ndarray, label: int) -> float:
    """
    根据模型 classes_ 顺序，取出对应 label 的概率。
    若拿不到 classes_，兜底假设 proba[1] 是 label=1，proba[0] 是 label=0。
    """
    if CLASSES is None:
        return float(proba_row[1] if label == 1 else proba_row[0])
    idx = CLASSES.index(label)
    return float(proba_row[idx])

def predict_with_proba(df_features: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("模型不支持 predict_proba()，请使用支持概率输出的分类器或Pipeline。")
    proba = model.predict_proba(df_features)
    pred = model.predict(df_features)
    # 返回：pred(0/1), p_uc, p_cd
    p_uc = np.array([proba_of_label(row, UC_LABEL) for row in proba], dtype=float)
    p_cd = np.array([proba_of_label(row, CD_LABEL) for row in proba], dtype=float)
    return pred.astype(int), p_uc, p_cd

def risk_hint_text(p_uc: float) -> str:
    if p_uc >= UC_HIGH_TH:
        return f"⚠️ 提示：UC 概率 ≥ {UC_HIGH_TH:.2f}，倾向 UC（建议结合临床进一步评估）"
    if UC_BORDER_LOW <= p_uc < UC_HIGH_TH:
        return f"ℹ️ 提示：UC 概率位于 {UC_BORDER_LOW:.2f}~{UC_HIGH_TH:.2f}，属于边界区，结果不确定性较高"
    return f"✅ 提示：UC 概率 < {UC_BORDER_LOW:.2f}，倾向 CD（建议结合临床进一步评估）"

# LIME 需要 numpy 输入 -> 转 DataFrame 再 predict_proba
def predict_proba_for_lime(x_np: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x_np, columns=FEATURES)
    return model.predict_proba(df)

lime_explainer = LimeTabularExplainer(
    training_data=X_bg.values,
    feature_names=FEATURES,
    class_names=["CD (0)", "UC (1)"],
    mode="classification",
)

# =========================
# 4. Tab1：单样本预测（医学范围优化）
# =========================
with tab1:
    st.subheader("单样本预测")

    # 常见医学范围（可按你数据实际分布再微调）
    # Age: 0-100 (步长1)
    # StoolFrequencyPerDay: 0-30 (步长1)
    # Platelets: 50-1000 (10^9/L) (步长1或5)
    # MCH: 15-40 (pg) (步长0.1)
    # WBC: 0.5-50 (10^9/L) (步长0.1)
    # HDL_Cholesterol: 0-150 (mg/dL) (步长1)

    st.sidebar.header("单样本输入（医学范围）")
    age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    stool = st.sidebar.number_input("StoolFrequencyPerDay (times/day)", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
    platelets = st.sidebar.number_input("Platelets (10^9/L)", min_value=50.0, max_value=1000.0, value=250.0, step=5.0)
    mch = st.sidebar.number_input("MCH (pg)", min_value=15.0, max_value=40.0, value=30.0, step=0.1, format="%.1f")
    wbc = st.sidebar.number_input("WBC (10^9/L)", min_value=0.5, max_value=50.0, value=7.0, step=0.1, format="%.1f")
    hdl = st.sidebar.number_input("HDL_Cholesterol (mg/dL)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)

    input_df = pd.DataFrame([{
        "Age": age,
        "StoolFrequencyPerDay": stool,
        "Platelets": platelets,
        "MCH": mch,
        "WBC": wbc,
        "HDL_Cholesterol": hdl,
    }])[FEATURES]

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🧾 输入数据")
        st.dataframe(input_df, use_container_width=True)

    if st.button("开始预测（单样本）"):
        pred, p_uc_arr, p_cd_arr = predict_with_proba(input_df)
        pred = int(pred[0])
        p_uc = float(p_uc_arr[0])
        p_cd = float(p_cd_arr[0])

        with c2:
            st.markdown("#### ✅ 预测结果")
            st.write(f"**预测类别：{'UC (1)' if pred == 1 else 'CD (0)'}**")
            st.write(f"UC 概率 P(UC=1)：**{p_uc:.4f}**")
            st.write(f"CD 概率 P(CD=0)：**{p_cd:.4f}**")

            # 阈值提示
            hint = risk_hint_text(p_uc)
            if p_uc >= UC_HIGH_TH:
                st.warning(hint)
            elif UC_BORDER_LOW <= p_uc < UC_HIGH_TH:
                st.info(hint)
            else:
                st.success(hint)

        st.markdown("#### 🔎 LIME 单样本解释（贡献最大的特征）")
        lime_exp = lime_explainer.explain_instance(
            data_row=input_df.values.flatten(),
            predict_fn=predict_proba_for_lime,
            num_features=len(FEATURES),
        )
        st.components.v1.html(lime_exp.as_html(show_table=True), height=600, scrolling=True)

# =========================
# 5. Tab2：批量CSV预测 + 下载 + 选行做LIME
# =========================
with tab2:
    st.subheader("批量CSV预测")
    st.markdown(
        f"""
请上传 CSV，至少包含以下列（列名必须一致）：
`{", ".join(FEATURES)}`
"""
    )

    uploaded = st.file_uploader("上传CSV文件", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"上传的CSV缺少列：{missing}")
            st.stop()

        df_feat = df[FEATURES].copy()
        before = df_feat.shape[0]
        df_feat = df_feat.dropna()
        dropped = before - df_feat.shape[0]
        if dropped > 0:
            st.warning(f"已自动丢弃包含缺失值的行：{dropped} 行")

        pred, p_uc, p_cd = predict_with_proba(df_feat)

        out = df.loc[df_feat.index].copy()  # 保留原始其它列（若有），并对齐索引
        out["P_UC"] = p_uc
        out["P_CD"] = p_cd
        out["Pred"] = pred
        out["PredLabel"] = np.where(out["Pred"] == 1, "UC", "CD")

        # 阈值分层提示
        def tier(x):
            if x >= UC_HIGH_TH:
                return "UC_high"
            if x >= UC_BORDER_LOW:
                return "borderline"
            return "CD_high"
        out["UC_Tier"] = out["P_UC"].apply(tier)

        # 展示概览
        st.markdown("#### 📊 预测概览")
        c1, c2, c3 = st.columns(3)
        c1.metric("总预测行数", f"{out.shape[0]}")
        c2.metric("预测为 UC (1)", f"{int((out['Pred'] == 1).sum())}")
        c3.metric("预测为 CD (0)", f"{int((out['Pred'] == 0).sum())}")

        st.markdown("#### 🧾 预测结果表（可滚动查看）")
        st.dataframe(out, use_container_width=True)

        # 下载
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 下载预测结果CSV",
            data=csv_bytes,
            file_name="uc_cd_predictions.csv",
            mime="text/csv",
        )

        # 选一行做 LIME
        st.markdown("#### 🔎 选择一行进行 LIME 解释")
        idx_list = list(out.index)
        selected_idx = st.selectbox("选择行索引（index）", idx_list)
        selected_row = out.loc[[selected_idx], FEATURES]  # DataFrame (1,6)

        st.write("选中样本特征：")
        st.dataframe(selected_row, use_container_width=True)

        if st.button("生成该行的 LIME 解释"):
            lime_exp2 = lime_explainer.explain_instance(
                data_row=selected_row.values.flatten(),
                predict_fn=predict_proba_for_lime,
                num_features=len(FEATURES),
            )
            st.components.v1.html(lime_exp2.as_html(show_table=True), height=600, scrolling=True)
