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
def make_unique(cols):
    seen = {}
    out = []
    for col in cols:
        base = col
        cnt = seen.get(base, 0)
        if cnt:
            col = f"{base}.{cnt}"
        seen[base] = cnt + 1
        out.append(col)
    return out

def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        tables = []
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                if len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    # 列名をトリムかつ一意化
                    cols = [str(c).strip() for c in df.columns]
                    df.columns = make_unique(cols)
                    tables.append(df)
    if not tables:
        return pd.DataFrame()
    # 列揃えして concat
    all_cols = sorted({c for df in tables for c in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in tables]
    return pd.concat(aligned, ignore_index=True, sort=False)

def fallback_ocr_pdf(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        try:
            s = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
            s.replace({"": pd.NA}, inplace=True)
            df[c] = pd.to_numeric(s, errors="ignore")
        except Exception:
            st.warning(f"⚠️ 列 '{c}' の正規化に失敗しました。")
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def reconcile_reports(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    base = pub.select_dtypes(include="number").sum().sum()
    rows = []
    for name, df in others.items():
        total = df.select_dtypes(include="number").sum().sum()
        if base != total:
            rows.append({
                "レポート": name,
                "公金日計合計": base,
                "他日報合計": total,
                "差異": total - base
            })
    return pd.DataFrame(rows)

def generate_ai_suggestions(df_diff: pd.DataFrame) -> str:
    table = df_diff.to_string(index=False)
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n\n"
        + table
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
        "- サイドバーでPDFをアップロード\n"
        "- 上部タブで「プレビュー」「差異サマリー」「AI示唆」を切り替え"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDFと他日報PDFをアップロード",
        type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFファイルをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    for f in uploaded:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        buf = f.read()
        if ext != ".pdf":
            st.sidebar.error(f"{name} はPDFではありません。")
            continue
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{name} のテーブル抽出失敗。OCRを表示。")
            st.sidebar.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
            st.sidebar.success(f"{name} を公金日計としてセット")
        else:
            others[name] = df
            st.sidebar.success(f"{name} を他日報としてセット")

    if pub_df.empty or not others:
        st.warning("公金日計PDFと他日報PDFをそれぞれ1件以上アップロードしてください。")
        return

    df_diff = reconcile_reports(pub_df, others)
    ai_text = generate_ai_suggestions(df_diff) if not df_diff.empty else ""

    tab1, tab2, tab3 = st.tabs(["🔍 プレビュー", "📊 差異サマリー", "🤖 AI示唆"])
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー: {name}")
            st.dataframe(df, use_container_width=True)

    with tab2:
        if df_diff.empty:
            st.success("差異は検出されませんでした。")
        else:
            st.subheader("■ 差異サマリー")
            st.dataframe(df_diff, use_container_width=True)

    with tab3:
        if not ai_text:
            st.info("差異のある列がないためAI示唆はありません。")
        else:
            st.subheader("■ 差異原因示唆 (Gemini 2.5)")
            st.markdown(ai_text)

if __name__ == "__main__":
    main()
