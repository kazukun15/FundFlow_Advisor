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

# ─── PDF ユーティリティ ─────────────────────────────────
def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        raw = []
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                if len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    # 列名トリム＆ユニーク化
                    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names([c.strip() for c in df.columns])
                    raw.append(df)
    if not raw:
        return pd.DataFrame()
    # 列をそろえて結合
    cols = sorted({c for df in raw for c in df.columns})
    aligned = [df.reindex(columns=cols) for df in raw]
    return pd.concat(aligned, ignore_index=True, sort=False)

def fallback_ocr_pdf(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 列名トリム
    df.columns = [str(c).strip() for c in df.columns]
    # 各セル → 文字列 → カンマ除去 → 空文字を NaN → 数値変換
    for c in df.columns:
        try:
            s = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
            s = s.replace({"": pd.NA})
            df[c] = pd.to_numeric(s, errors="ignore")
        except Exception:
            st.warning(f"⚠️ 列 '{c}' の正規化に失敗しました。")
    # 完全空行・空列を削除
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def reconcile_reports(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    """
    公金日計(pub) と他日報(others)のすべての数値列について、
    列ごとに (pub_sum, other_sum, diff) を計算しテーブル化。
    """
    rows = []
    # pubの数値列
    pub_nums = pub.select_dtypes(include="number")
    for name, df in others.items():
        other_nums = df.select_dtypes(include="number")
        # 比較対象となる数値列のユニオン
        all_cols = sorted(set(pub_nums.columns) | set(other_nums.columns))
        for col in all_cols:
            base = pub_nums[col].sum() if col in pub_nums else 0
            comp = other_nums[col].sum() if col in other_nums else 0
            rows.append({
                "レポート": name,
                "列名": col,
                "公金日計合計": base,
                "他日報合計": comp,
                "差異": comp - base
            })
    return pd.DataFrame(rows)

def generate_ai_suggestions(df_diff: pd.DataFrame) -> str:
    table = df_diff[df_diff["差異"] != 0].to_string(index=False)
    prompt = (
        "以下の日報突合結果のうち、差異のある列について、\n"
        "なぜ差異が発生したか可能性を箇条書きで示してください。\n\n"
        + table
    )
    resp = genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author":"user","content":prompt}],
        temperature=0.7
    )
    return resp.candidates[0].message.content

# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.title("FundFlow Advisor 🏦")
    st.markdown(
        "- サイドバーで **PDF** ファイルをアップロード\n"
        "- 上部タブで「プレビュー」「差異サマリー」「AI示唆」を切り替え\n"
        "- 差異サマリーでは列ごとの集計結果をすべて表示します"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF をアップロード",
        type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFファイルを選択してください。")
        return

    pub_df, others = pd.DataFrame(), {}
    for f in uploaded:
        name, ext = f.name, os.path.splitext(f.name)[1].lower()
        buf = f.read()
        if ext != ".pdf":
            st.sidebar.error(f"{name} はPDF形式ではありません。スキップします。")
            continue
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{name} のテーブル抽出失敗。OCR結果を表示します。")
            st.sidebar.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=200)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
            st.sidebar.success(f"基準(Pub) として {name} を設定")
        else:
            others[name] = df
            st.sidebar.success(f"他日報として {name} を読み込み")

    if pub_df.empty or not others:
        st.warning("公金日計と他日報が揃いません。PDFを追加でアップロードしてください。")
        return

    # 突合
    df_diff = reconcile_reports(pub_df, others)

    # AI示唆用テキスト
    ai_text = ""
    if not df_diff[df_diff["差異"] != 0].empty:
        ai_text = generate_ai_suggestions(df_diff)

    # タブ表示
    tab1, tab2, tab3 = st.tabs(["🔍 プレビュー", "📊 差異サマリー", "🤖 AI示唆"])
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー：{name}")
            st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("■ 列ごとの突合結果")
        st.dataframe(df_diff, use_container_width=True)

    with tab3:
        if not ai_text:
            st.info("差異のある列がないため、AI示唆はありません。")
        else:
            st.subheader("■ Gemini 2.5 による差異原因示唆")
            st.markdown(ai_text)

if __name__ == "__main__":
    main()
