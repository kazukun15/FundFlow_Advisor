import io
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── Google API Key 設定（Secrets） ────────────────────────────
api_key = st.secrets.get("google", {}).get("api_key")
if not api_key:
    st.error("❌ .streamlit/secrets.toml の [google] api_key を設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ─── PDF解析関数 ───────────────────────────────────────────
def extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = [tbl for page in pdf.pages for tbl in (page.extract_tables() or [])]
    dfs = [pd.DataFrame(tbl[1:], columns=tbl[0]) for tbl in tables if len(tbl) > 1]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def fallback_ocr_pdf(file_bytes: bytes) -> str:
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="jpn")
    return text

# ─── データ正規化関数 ───────────────────────────────────
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", "").str.strip()
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

# ─── 突合ロジック ───────────────────────────────────
def reconcile_reports(pub_df: pd.DataFrame, other_dfs: dict) -> list[dict]:
    pub_sum = (
        pub_df["金額"].sum()
        if "金額" in pub_df.columns
        else pub_df.select_dtypes(include="number").sum().sum()
    )
    results = []
    for name, df in other_dfs.items():
        df_sum = (
            df["金額"].sum()
            if "金額" in df.columns
            else df.select_dtypes(include="number").sum().sum()
        )
        if pub_sum != df_sum:
            results.append({
                "レポート": name,
                "公金日計合計": pub_sum,
                "他日報合計": df_sum,
                "差異": df_sum - pub_sum
            })
    return results

# ─── AI示唆生成関数 ───────────────────────────────────
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

# ─── Streamlit UI ───────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor", layout="wide")
    st.title("FundFlow Advisor")
    st.markdown(
        "公金日計PDFと各部署日報PDFをアップロードし、"
        "差異突合結果と Gemini 2.5 による原因示唆を行います。"
    )

    # type制限なしでまずアップロード
    pub_file = st.file_uploader("📑 公金日計PDFをアップロード", type=None)
    other_files = st.file_uploader(
        "📑 他部署日報PDFをアップロード（複数可）", 
        type=None, 
        accept_multiple_files=True
    )

    # 入力チェック
    if not pub_file or not other_files:
        st.info("まず両方のPDFをアップロードしてください。")
        return

    # ファイル名拡張子チェック
    if not pub_file.name.lower().endswith(".pdf"):
        st.error("公金日計ファイルはPDF形式(.pdf)をアップロードしてください。")
        return
    for f in other_files:
        if not f.name.lower().endswith(".pdf"):
            st.error(f"「{f.name}」はPDF形式(.pdf)ではありません。")
            return

    # 公金日計解析
    pub_bytes = pub_file.read()
    df_pub = extract_tables_from_pdf(pub_bytes)
    if df_pub.empty:
        st.warning("公金日計のテーブル抽出に失敗しました。OCR結果をご確認ください。")
        st.text_area("OCR（公金日計）", fallback_ocr_pdf(pub_bytes), height=200)
    df_pub = normalize_df(df_pub)
    st.subheader("公金日計プレビュー")
    st.dataframe(df_pub)

    # 他部署日報解析
    other_dfs = {}
    for f in other_files:
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.warning(f"{f.name} の抽出に失敗。OCR結果を表示します。")
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
