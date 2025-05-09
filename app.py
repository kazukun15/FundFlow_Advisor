import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import io
import openai

# ─── 設定 ───────────────────────────────────────────
openai.api_key = st.secrets["openai"]["api_key"]

# ─── PDF抽出関数 ────────────────────────────────────
def extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    """PDFからテーブルを抽出してDataFrameとして返す"""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    dfs = []
    for table in tables:
        if table and len(table) > 1:
            df = pd.DataFrame(table[1:], columns=table[0])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def fallback_ocr_pdf(file_bytes: bytes) -> str:
    """OCRでPDFを画像化し、テキストを抽出"""
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='jpn')
    return text

# ─── データ正規化関数 ────────────────────────────────────
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """カンマ除去・数値変換を試みるクリーン処理"""
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

# ─── 突合関数 ──────────────────────────────────────
def reconcile_reports(pub_df: pd.DataFrame, other_dfs: dict) -> list:
    """公金日計と他日報を比較し、差異をまとめる"""
    pub_sum = pub_df['金額'].sum() if '金額' in pub_df.columns else pub_df.select_dtypes(include='number').sum().sum()
    suggestions = []
    for name, df in other_dfs.items():
        df_sum = df['金額'].sum() if '金額' in df.columns else df.select_dtypes(include='number').sum().sum()
        if pub_sum != df_sum:
            suggestions.append({
                'レポート': name,
                '公金日計合計': pub_sum,
                '他日報合計': df_sum,
                '差異': df_sum - pub_sum
            })
    return suggestions

# ─── AI示唆生成関数 ─────────────────────────────────
def generate_ai_suggestions(suggestions: list) -> str:
    """差異情報をもとにGemini 2.5で原因示唆を生成"""
    prompt = (
        "以下の日報突合結果について、差異の原因を箇条書きで示してください。\n"
        f"{suggestions}"
    )
    response = openai.ChatCompletion.create(
        model='gemini-2.5',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ─── Streamlit UI ───────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor", layout="wide")
    st.title("FundFlow Advisor")
    st.markdown("公金日計PDFと他部署日報PDFをアップロードし、突合結果と原因示唆を提供します。")

    pub_file = st.file_uploader("📑 公金日計PDFをアップロード", type='pdf')
    other_files = st.file_uploader("📑 他部署日報PDFをアップロード（複数可）", type='pdf', accept_multiple_files=True)

    if not pub_file or not other_files:
        st.info("まずはすべてのPDFをアップロードしてください。")
        return

    # 公金日計解析
    pub_bytes = pub_file.read()
    pub_df = extract_tables_from_pdf(pub_bytes)
    if pub_df.empty:
        st.warning("テーブル抽出に失敗しました。OCR結果を確認してください。")
        st.text_area("OCRテキスト（公金日計）", fallback_ocr_pdf(pub_bytes), height=200)
    pub_df = normalize_df(pub_df)
    st.subheader("公金日計プレビュー")
    st.dataframe(pub_df)

    # 他部署日報解析
    other_dfs = {}
    for f in other_files:
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.warning(f"{f.name} 抽出失敗。OCR結果を表示します。")
            st.text_area(f"OCRテキスト（{f.name}）", fallback_ocr_pdf(buf), height=200)
        df = normalize_df(df)
        other_dfs[f.name] = df
        st.subheader(f"{f.name} プレビュー")
        st.dataframe(df)

    # 突合実行
    suggestions = reconcile_reports(pub_df, other_dfs)
    if suggestions:
        st.subheader("▶ 差異サマリ")
        st.table(pd.DataFrame(suggestions))
        st.subheader("▶ AIによる原因示唆（Gemini 2.5）")
        ai_text = generate_ai_suggestions(suggestions)
        st.markdown(ai_text)
    else:
        st.success("差異は検出されませんでした。")

if __name__ == "__main__":
    main()

