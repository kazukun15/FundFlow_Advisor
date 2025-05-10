import io
import os
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
    st.error("❌ `.streamlit/secrets.toml` に [google] api_key を設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ─── ユーティリティ ─────────────────────────────────────
def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    """
    pdfplumberで抽出した複数テーブルを
    列名をそろえてから結合。違う列はNaN埋め。
    """
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        raw_tables = []
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    # 列名を文字列化
                    df.columns = [str(c).strip() for c in df.columns]
                    raw_tables.append(df)
    if not raw_tables:
        return pd.DataFrame()
    # 全テーブルの列をユニオンしてから再構成
    all_cols = sorted({c for df in raw_tables for c in df.columns})
    aligned = []
    for df in raw_tables:
        aligned.append(df.reindex(columns=all_cols))
    return pd.concat(aligned, ignore_index=True, sort=False)

def fallback_ocr_pdf(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 余計な空白・カンマ除去
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        try:
            df[c] = (
                df[c]
                .astype(str)
                .map(lambda x: str(x).replace(",", "").strip())
                .replace({"": pd.NA})
            )
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            st.warning(f"⚠️ 列 '{c}' の正規化に失敗、元のまま保持します。")
    # 完全空行・空列は削除
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def reconcile_reports(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    base = pub.select_dtypes(include="number").sum().sum()
    rows = []
    for name, df in others.items():
        s = df.select_dtypes(include="number").sum().sum()
        if base != s:
            rows.append({
                "レポート": name,
                "公金日計合計": base,
                "他日報合計": s,
                "差異": s - base
            })
    return pd.DataFrame(rows)

def generate_ai_suggestions(df_diff: pd.DataFrame) -> str:
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n\n"
        + df_diff.to_string(index=False)
    )
    resp = genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.candidates[0].message.content

# ─── メイン ───────────────────────────────────────────
def main():
    st.title("FundFlow Advisor 🏦")
    st.markdown(
        "- PDFだけをアップロードして処理します\n"
        "- 上部タブで「プレビュー」「差異サマリー」「AI示唆」を切り替え"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF をアップロード",
        type=["pdf"],
        accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFファイルをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}

    for f in uploaded:
        name = f.name
        buf = f.read()
        # テーブル抽出
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{name} からテーブルを抽出できません。OCR結果を表示します。")
            st.sidebar.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
        else:
            others[name] = df

    if pub_df.empty or not others:
        st.warning("公金日計PDF と 他日報PDF の両方が必要です。")
        return

    df_diff = reconcile_reports(pub_df, others)
    ai_text = generate_ai_suggestions(df_diff) if not df_diff.empty else ""

    tab1, tab2, tab3 = st.tabs(["🔍 プレビュー", "📊 差異サマリー", "🤖 AI示唆"])
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.code(pub_df.to_string(index=False), language="")
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー：{name}")
            st.code(df.to_string(index=False), language="")

    with tab2:
        if df_diff.empty:
            st.success("差異は検出されませんでした。")
        else:
            st.subheader("■ 差異サマリー")
            st.code(df_diff.to_string(index=False), language="")

    with tab3:
        if df_diff.empty:
            st.info("差異がないためAI示唆はありません。")
        else:
            st.subheader("■ 差異原因示唆 (Gemini 2.5)")
            st.markdown(ai_text)

if __name__ == "__main__":
    main()
