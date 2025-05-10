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

# ─── ユーティリティ関数 ─────────────────────────────────
def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        tables = [t for p in pdf.pages for t in (p.extract_tables() or [])]
    dfs = [pd.DataFrame(t[1:], columns=t[0]) for t in tables if len(t) > 1]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def fallback_ocr_pdf(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        try:
            def clean_val(x):
                y = str(x).replace(",", "").strip()
                return "" if y.lower() == "none" else y
            cleaned = df[c].map(clean_val)
            df[c] = pd.to_numeric(cleaned, errors="ignore")
        except Exception:
            st.warning(f"⚠️ 列 '{c}' の正規化に失敗。元のまま保持します。")
    return df

def clean_df_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # 空文字を NA に置換し、空行・空列を削除
    df2.replace("", pd.NA, inplace=True)
    df2.dropna(axis=0, how="all", inplace=True)
    df2.dropna(axis=1, how="all", inplace=True)
    return df2

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
    table_str = df_diff.to_string(index=False)
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n\n"
        + table_str
    )
    resp = genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author":"user","content":prompt}],
        temperature=0.7
    )
    return resp.candidates[0].message.content

# ─── メイン ───────────────────────────────────────────
def main():
    st.title("FundFlow Advisor 🏦")
    st.markdown(
        "- サイドバーで **PDF** / **Excel** ファイルをアップロード\n"
        "- 上部タブで **プレビュー** / **差異サマリー** / **AI示唆** を切り替え"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 ファイルをアップロード (PDF/XLS/XLSX)",
        type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでファイルをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    allowed = {".pdf", ".xls", ".xlsx"}

    for f in uploaded:
        name, ext = f.name, os.path.splitext(f.name)[1].lower()
        buf = f.read()
        if ext not in allowed:
            st.sidebar.error(f"{name} は非対応形式です。")
            continue

        if ext == ".pdf":
            df = extract_tables_from_pdf(buf)
            if df.empty:
                st.sidebar.warning(f"{name} の抽出に失敗。OCRを表示。")
                st.sidebar.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=150)
                df = pd.DataFrame()
            df = normalize_df(df)
            if pub_df.empty:
                pub_df = df
            else:
                others[name] = df

        else:
            engine = "xlrd" if ext == ".xls" else "openpyxl"
            try:
                sheets = pd.read_excel(io.BytesIO(buf), sheet_name=None, engine=engine)
            except Exception as e:
                st.sidebar.error(f"{name} 読込エラー: {e}")
                continue
            for sheet, sdf in sheets.items():
                key = f"{name}:{sheet}"
                others[key] = normalize_df(sdf)

    if pub_df.empty or not others:
        st.warning("公金日計(PDF) と 他日報 をそれぞれ1件以上含めてください。")
        return

    df_diff = reconcile_reports(pub_df, others)
    ai_text = generate_ai_suggestions(df_diff) if not df_diff.empty else ""

    tab1, tab2, tab3 = st.tabs(["🔍 プレビュー", "📊 差異サマリー", "🤖 AI示唆"])
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        clean_pub = clean_df_for_preview(pub_df)
        st.code(clean_pub.head(10).to_string(index=False), language="")
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー: {name}")
            clean_df = clean_df_for_preview(df)
            st.code(clean_df.head(10).to_string(index=False), language="")

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
