import io, os, re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── 設定 ─────────────────────────────────────────────
st.set_page_config(page_title="FundFlow Advisor", layout="wide")
api_key = st.secrets.get("google", {}).get("api_key")
if not api_key:
    st.error("❌ Secrets に google.api_key を設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ─── ユーティリティ ─────────────────────────────────────
def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        cnt = seen.get(c, 0)
        name = f"{c}.{cnt}" if cnt else c
        seen[c] = cnt + 1
        out.append(name)
    return out

def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        tables = []
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                if len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    cols = [str(c).strip() for c in df.columns]
                    df.columns = make_unique(cols)
                    tables.append(df)
    if not tables:
        return pd.DataFrame()
    all_cols = sorted({c for df in tables for c in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in tables]
    return pd.concat(aligned, ignore_index=True, sort=False)

def fallback_ocr_pdf(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    for c in df.columns:
        s = (
            df[c]
            .astype(str)
            .map(lambda x: re.sub(r"[^\d\.\-]", "", x))
            .replace({"": pd.NA})
        )
        df[c] = pd.to_numeric(s, errors="ignore")
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def reconcile_reports(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    base = pub.select_dtypes(include="number").sum().sum()
    rows = []
    for name, df in others.items():
        total = df.select_dtypes(include="number").sum().sum()
        rows.append({
            "レポート": name,
            "公金日計合計": base,
            "他日報合計": total,
            "差異": total - base
        })
    return pd.DataFrame(rows)

def analyze_cash_flow(pub: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    ob = pub.get("会計前日残高", pd.Series(dtype=float)).dropna()
    ob = ob.iloc[0] if not ob.empty else None
    nums = pub.select_dtypes(include="number")
    col_sums = nums.sum().rename("合計").to_frame()
    if "入金" in nums.columns and "出金" in nums.columns:
        inflow = nums["入金"].sum()
        outflow = nums["出金"].sum()
    elif "収入" in nums.columns and "支出" in nums.columns:
        inflow = nums["収入"].sum()
        outflow = nums["支出"].sum()
    else:
        inflow = nums[nums > 0].sum().sum()
        outflow = -nums[nums < 0].sum().sum()
    net = inflow - outflow
    metrics = {"前日残高": ob, "総流入": inflow, "総流出": outflow, "純増減": net}
    return metrics, col_sums

def generate_ai_suggestions(df_diff: pd.DataFrame) -> str:
    # 差異のある行のみテキスト化
    diff_rows = df_diff[df_diff["差異"] != 0]
    text = diff_rows.to_string(index=False) if not diff_rows.empty else "（全レポート差異なし）"
    prompt = (
        "以下の差異について、原因を箇条書きで示してください。\n\n" + text
    )
    # 正しい呼び出しに修正
    response = genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author": "user", "content": prompt}],
        temperature=0.7
    )
    return response.candidates[0].message.content

# ─── UI ─────────────────────────────────────────────────
def main():
    st.title("FundFlow Advisor 🏦")
    st.markdown(
        "- サイドバーでPDFをアップロード\n"
        "- 上部タブで「プレビュー」「差異サマリー」「分析」「AI示唆」を切り替え"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF", type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    for f in uploaded:
        if not f.name.lower().endswith(".pdf"):
            st.sidebar.error(f"{f.name} はPDFではありません")
            continue
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{f.name} の抽出失敗、OCRを表示")
            st.sidebar.text_area(f"OCR({f.name})", fallback_ocr_pdf(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
            st.sidebar.success(f"{f.name} を基準に設定")
        else:
            others[f.name] = df
            st.sidebar.success(f"{f.name} を他日報に追加")

    if pub_df.empty or not others:
        st.warning("基準と他日報のPDFをそれぞれアップしてください。")
        return

    df_diff = reconcile_reports(pub_df, others)
    cash_metrics, col_sums = analyze_cash_flow(pub_df)
    ai_text = generate_ai_suggestions(df_diff)

    tabs = st.tabs(["🔍 プレビュー", "📊 差異サマリー", "💡 分析", "🤖 AI示唆"])
    with tabs[0]:
        st.subheader("■ 公金日計プレビュー")
        st.write("**列名**", list(pub_df.columns))
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー：{name}")
            st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("■ 差異サマリー")
        st.dataframe(df_diff, use_container_width=True)

    with tabs[2]:
        st.subheader("■ キャッシュフロー分析")
        if cash_metrics["前日残高"] is not None:
            st.metric("前日残高", f"{int(cash_metrics['前日残高']):,}")
        c1, c2, c3 = st.columns(3)
        c1.metric("総流入", f"{int(cash_metrics['総流入']):,}")
        c2.metric("総流出", f"{int(cash_metrics['総流出']):,}")
        c3.metric("純増減", f"{int(cash_metrics['純増減']):,}")
        st.subheader("◾ 列ごとの合計値")
        st.dataframe(col_sums, use_container_width=True)
        if cash_metrics["純増減"] < 0:
            st.error("⚠️ 資金ショートリスクがあります。")
        elif cash_metrics["純増減"] < cash_metrics["総流出"] * 0.1:
            st.warning("⚠️ 資金余裕が乏しい可能性があります。")
        else:
            st.success("✅ キャッシュポジションは健全です。")

    with tabs[3]:
        st.subheader("■ 差異原因示唆 (Gemini 2.5)")
        st.markdown(ai_text)

if __name__ == "__main__":
    main()
