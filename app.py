import io
import os
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ──────────────────────────────────────────────────
# Google API Key設定 (Secrets経由)
api_key = st.secrets.get("google", {}).get("api_key")
if not api_key:
    st.error("❌ `.streamlit/secrets.toml`に[google] api_keyを設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ──────────────────────────────────────────────────
# PDFからテーブルを抽出
def extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = [tbl for page in pdf.pages for tbl in (page.extract_tables() or [])]
    dfs = [pd.DataFrame(tbl[1:], columns=tbl[0]) for tbl in tables if len(tbl) > 1]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# PDF OCR
def fallback_ocr_pdf(file_bytes: bytes) -> str:
    images = convert_from_bytes(file_bytes)
    return "".join(pytesseract.image_to_string(img, lang="jpn") for img in images)

# ──────────────────────────────────────────────────
# 頑強なデータ正規化関数（最終改善版）
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        try:
            # 一貫した文字列化処理で安全に
            s = df[col].astype(str).map(lambda x: x.replace(",", "").strip())
            # 数値に安全変換
            df[col] = pd.to_numeric(s, errors="ignore")
        except Exception as e:
            raise ValueError(f"列 '{col}' でエラー発生。詳細: {e}")
    return df

# 表示サニタイズ
def sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x if isinstance(x, (str,int,float,bool,type(None))) else str(x))
    return df

# 突合処理
def reconcile_reports(pub_df: pd.DataFrame, other_dfs: dict) -> list[dict]:
    pub_sum = pub_df.select_dtypes(include="number").sum().sum()
    results = []
    for name, df in other_dfs.items():
        df_sum = df.select_dtypes(include="number").sum().sum()
        if pub_sum != df_sum:
            results.append({
                "レポート": name,
                "公金日計合計": pub_sum,
                "他日報合計": df_sum,
                "差異": df_sum - pub_sum
            })
    return results

# AI 示唆生成
def generate_ai_suggestions(suggestions: list[dict]) -> str:
    df = pd.DataFrame(suggestions)
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n\n"
        + df.to_markdown(index=False)
    )
    response = genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author": "user", "content": prompt}],
        temperature=0.7
    )
    return response.candidates[0].message.content

# ──────────────────────────────────────────────────
# Streamlit UI
def main():
    st.set_page_config(page_title="FundFlow Advisor", layout="wide")
    st.title("FundFlow Advisor 📊✨")
    st.markdown(
        "PDF/Excelをアップロードし、日報突合とGemini2.5による差異原因示唆を行います。"
    )

    uploaded_files = st.file_uploader(
        "📁 PDFまたはExcelファイルをアップロード",
        type=["pdf", "xls", "xlsx"],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("まずファイルをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    other_dfs = {}

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        ext = os.path.splitext(name)[1].lower()
        buf = uploaded_file.read()

        if ext == ".pdf":
            df = extract_tables_from_pdf(buf)
            if df.empty:
                st.warning(f"[PDF] {name} はOCRが必要です。")
                st.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=200)
            df = normalize_df(df)
            if pub_df.empty:
                pub_df = df
                st.subheader(f"公金日計プレビュー({name})")
                st.dataframe(sanitize_df_for_display(pub_df))
            else:
                other_dfs[name] = df
                with st.expander(f"他日報プレビュー({name})"):
                    st.dataframe(sanitize_df_for_display(df))

        elif ext in [".xls", ".xlsx"]:
            engine = "xlrd" if ext == ".xls" else "openpyxl"
            try:
                sheets = pd.read_excel(io.BytesIO(buf), sheet_name=None, engine=engine)
            except Exception as e:
                st.error(f"{name} 読込エラー: {e}")
                continue
            for sheet_name, sheet_df in sheets.items():
                key = f"{name}:{sheet_name}"
                try:
                    df_clean = normalize_df(sheet_df)
                    other_dfs[key] = df_clean
                    with st.expander(f"Excelシート({key})"):
                        st.dataframe(sanitize_df_for_display(df_clean))
                except Exception as e:
                    st.error(f"{key} 正規化エラー: {e}")

    if pub_df.empty or not other_dfs:
        st.warning("公金日計と他日報が揃っていません。")
        return

    diffs = reconcile_reports(pub_df, other_dfs)
    if diffs:
        st.subheader("🚩 差異サマリー")
        st.table(sanitize_df_for_display(pd.DataFrame(diffs)))
        st.subheader("🤖 Gemini 2.5による差異の原因示唆")
        suggestion = generate_ai_suggestions(diffs)
        st.markdown(suggestion)
    else:
        st.success("🎉 差異はありませんでした！")

if __name__ == "__main__":
    main()
