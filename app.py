import io, os
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── Google API Key 設定（Secrets）────────────────────────────
api_key = st.secrets.get("google", {}).get("api_key")
if not api_key:
    st.error("❌ `.streamlit/secrets.toml` の [google] api_key を設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ─── PDF解析 ────────────────────────────────────────────
def extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = [tbl for page in pdf.pages for tbl in (page.extract_tables() or [])]
    dfs = [pd.DataFrame(tbl[1:], columns=tbl[0]) for tbl in tables if len(tbl) > 1]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def fallback_ocr_pdf(file_bytes: bytes) -> str:
    images = convert_from_bytes(file_bytes)
    return "".join(pytesseract.image_to_string(img, lang="jpn") for img in images)

# ─── データ正規化（安全版） ─────────────────────────────────
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    ・列名をすべて文字列化して strip
    ・各セルを map() で文字列化→カンマ除去→trim→数値変換
    """
    df = df.copy()
    # 列名を str 化して strip
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        # map で要素単位に処理
        cleaned = df[col].map(lambda x: str(x).replace(",", "").strip())
        # 数値変換（できないものは元の文字列）
        df[col] = pd.to_numeric(cleaned, errors="ignore")
    return df

# ─── 表示サニタイズ ────────────────────────────────────
def sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x if isinstance(x, (str, int, float, bool, type(None))) else str(x))
    return df

# ─── 突合ロジック ─────────────────────────────────────
def reconcile_reports(pub_df: pd.DataFrame, other_dfs: dict) -> list[dict]:
    pub_sum = (
        pub_df.get("金額", pd.Series(dtype=float)).sum()
        if "金額" in pub_df.columns
        else pub_df.select_dtypes(include="number").sum().sum()
    )
    results = []
    for name, df in other_dfs.items():
        df_sum = (
            df.get("金額", pd.Series(dtype=float)).sum()
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

# ─── AI示唆生成 ───────────────────────────────────
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

# ─── Streamlit UI ────────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor", layout="wide")
    st.title("FundFlow Advisor")
    st.markdown(
        "PDF または Excel をアップロードし、日計の突合および "
        "Gemini 2.5 での原因示唆を行います。"
    )

    uploaded = st.file_uploader(
        "📁 ファイルをアップロード（PDF / XLS / XLSX）",
        type=None,
        accept_multiple_files=True
    )
    if not uploaded:
        st.info("まずはファイルをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    other_dfs = {}

    for f in uploaded:
        name, ext = f.name, os.path.splitext(f.name)[1].lower()
        buf = f.read()

        if ext == ".pdf":
            df = extract_tables_from_pdf(buf)
            if df.empty:
                st.warning(f"[PDF] {name} の抽出に失敗。OCR結果を表示します。")
                st.text_area(f"OCR（{name}）", fallback_ocr_pdf(buf), height=200)
                df = pd.DataFrame()
            df = normalize_df(df)
            if pub_df.empty:
                pub_df = df
                st.subheader("公金日計プレビュー")
                st.dataframe(sanitize_df_for_display(pub_df))
            else:
                other_dfs[name] = df
                with st.expander(f"他日報プレビュー：{name}", expanded=False):
                    st.dataframe(sanitize_df_for_display(df))

        elif ext in (".xls", ".xlsx"):
            try:
                sheets = pd.read_excel(io.BytesIO(buf), sheet_name=None,
                                       engine="xlrd" if ext == ".xls" else None)
            except Exception as e:
                st.error(f"{name} の Excel 読み込みエラー: {e}")
                continue
            for sheet_name, sheet_df in sheets.items():
                key = f"{name}:{sheet_name}"
                df_clean = normalize_df(sheet_df)
                other_dfs[key] = df_clean
                with st.expander(f"プレビュー：{key}", expanded=False):
                    st.dataframe(sanitize_df_for_display(df_clean))

        else:
            st.error(f"{name} はサポート外の拡張子です。")

    if pub_df.empty or not other_dfs:
        st.warning("公金日計と他日報の両方が必要です。")
        return

    diffs = reconcile_reports(pub_df, other_dfs)
    if diffs:
        st.subheader("▶ 差異サマリ")
        st.table(sanitize_df_for_display(pd.DataFrame(diffs)))
        st.subheader("▶ Gemini 2.5 による原因示唆")
        st.markdown(generate_ai_suggestions(diffs))
    else:
        st.success("🎉 差異は検出されませんでした。")

if __name__ == "__main__":
    main()
