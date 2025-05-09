import os
import io

import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import openai

# ─── OpenAI API Key 設定 ──────────────────────────────────
# ① secrets.toml の [openai] セクション
# ② 環境変数 OPENAI_API_KEY
# ③ 起動時のテキスト入力フォーム
api_key = None
# secrets.toml に "openai": {"api_key": "..."} がある場合
if st.secrets.get("openai", {}).get("api_key"):
    api_key = st.secrets["openai"]["api_key"]
# 次に環境変数
elif os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")
# 最後にユーザー入力
else:
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="secrets.toml または環境変数に設定がなければこちらに入力"
    )

if not api_key:
    st.error("❌ OpenAI API Key が設定されていません。")
    st.stop()

openai.api_key = api_key

# ─── PDF抽出関数 ─────────────────────────────────────────
def extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables() or [])
    dfs = []
    for table in tables:
        if table and len(table) > 1:
            df = pd.DataFrame(table[1:], columns=table[0])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def fallback_ocr_pdf(file_bytes: bytes) -> str:
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='jpn')
    return text

# ─── データ正規化関数 ────────────────────────────────────
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(',', '').str.strip()
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

# ─── 突合関数 ───────────────────────────────────────────
def reconcile_reports(pub_df: pd.DataFrame, other_dfs: dict) -> list[dict]:
    pub_sum = (
        pub_df['金額'].sum()
        if '金額' in pub_df.columns
        else pub_df.select_dtypes(include='number').sum().sum()
    )
    results = []
    for name, df in other_dfs.items():
        df_sum = (
            df['金額'].sum()
            if '金額' in df.columns
            else df.select_dtypes(include='number').sum().sum()
        )
        if pub_sum != df_sum:
            results.append({
                'レポート': name,
                '公金日計合計': pub_sum,
                '他日報合計': df_sum,
                '差異': df_sum - pub_sum
            })
    return results

# ─── AI示唆生成関数 ────────────────────────────────────
def generate_ai_suggestions(suggestions: list[dict]) -> str:
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n\n"
        + pd.DataFrame(suggestions).to_markdown(index=False)
    )
    resp = openai.ChatCompletion.create(
        model='gemini-2.5',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor", layout="wide")
    st.title("FundFlow Advisor")
    st.markdown("公金日計PDFと各部署日報PDFをアップロードして突合し、Gemini 2.5で原因示唆まで行うアプリです。")

    pub_file = st.file_uploader("📑 公金日計PDFをアップロード", type='pdf')
    other_files = st.file_uploader(
        "📑 他部署日報PDFをアップロード（複数可）",
        type='pdf',
        accept_multiple_files=True
    )

    if not pub_file or not other_files:
        st.info("まずは両方のPDFをアップロードしてください。")
        return

    # 公金日計解析
    buf = pub_file.read()
    df_pub = extract_tables_from_pdf(buf)
    if df_pub.empty:
        st.warning("公金日計のテーブル抽出に失敗しました。OCR結果をご確認ください。")
        st.text_area("OCR（公金日計）", fallback_ocr_pdf(buf), height=200)
    df_pub = normalize_df(df_pub)
    st.subheader("公金日計プレビュー")
    st.dataframe(df_pub)

    # 他部署日報解析
    other_dfs = {}
    for f in other_files:
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.warning(f"{f.name} のテーブル抽出に失敗しました。OCR結果を表示します。")
            st.text_area(f"OCR（{f.name}）", fallback_ocr_pdf(buf), height=200)
        df = normalize_df(df)
        other_dfs[f.name] = df
        st.subheader(f"{f.name} プレビュー")
        st.dataframe(df)

    # 突合＆AI示唆
    diffs = reconcile_reports(df_pub, other_dfs)
    if diffs:
        st.subheader("▶ 差異サマリ")
        st.table(pd.DataFrame(diffs))
        st.subheader("▶ Gemini 2.5 による原因示唆")
        suggestion = generate_ai_suggestions(diffs)
        st.markdown(suggestion)
    else:
        st.success("🎉 差異は検出されませんでした。")

if __name__ == "__main__":
    main()
