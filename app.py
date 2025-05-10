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
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        try:
            s = (df[c]
                 .astype(str)
                 .str.replace(",", "", regex=False)
                 .str.strip()
                 .replace({"": pd.NA}))
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

def analyze_cash_flow(pub: pd.DataFrame) -> dict:
    # 金額列を特定
    if "金額" in pub.columns:
        amt = pub["金額"].dropna()
    else:
        nums = pub.select_dtypes(include="number")
        if nums.empty:
            return {}
        amt = nums.iloc[:, 0]
    inflow = amt[amt > 0].sum()
    outflow = -amt[amt < 0].sum()
    net = inflow - outflow
    return {"総流入": inflow, "総流出": outflow, "純増減": net}

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

# ─── UI ─────────────────────────────────────────────────
def main():
    st.title("FundFlow Advisor 🏦")
    st.markdown(
        "- サイドバーで **PDF** ファイルをアップロード\n"
        "- 上部タブで「プレビュー」「差異サマリー」「分析」「AI示唆」を切り替え"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF を選択",
        type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    for f in uploaded:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        buf = f.read()
        if ext != ".pdf":
            st.sidebar.error(f"{name} はPDFではありません。スキップ。")
            continue
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{name} の抽出失敗。OCRを表示。")
            st.sidebar.text_area(f"OCR({name})", fallback_ocr_pdf(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
            st.sidebar.success(f"{name} を基準データに設定")
        else:
            others[name] = df
            st.sidebar.success(f"{name} を他日報に追加")

    if pub_df.empty or not others:
        st.warning("公金日計PDFと他日報PDFを最低1件ずつアップロードしてください。")
        return

    df_diff = reconcile_reports(pub_df, others)
    cash_metrics = analyze_cash_flow(pub_df)
    ai_text = generate_ai_suggestions(df_diff) if not df_diff.empty else ""

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 プレビュー", "📊 差異サマリー", "💡 分析", "🤖 AI示唆"]
    )

    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー：{name}")
            st.dataframe(df, use_container_width=True)

    with tab2:
        if df_diff.empty:
            st.success("差異は検出されませんでした。")
        else:
            st.subheader("■ 差異サマリー")
            st.dataframe(df_diff, use_container_width=True)

    with tab3:
        st.subheader("■ 多角的キャッシュフロー分析")
        if cash_metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("総流入", f"{cash_metrics['総流入']:,}")
            col2.metric("総流出", f"{cash_metrics['総流出']:,}")
            col3.metric("純増減", f"{cash_metrics['純増減']:,}")
            # リスク評価
            if cash_metrics["純増減"] < 0:
                st.error("⚠️ 資金ショートのリスクがあります。")
            elif cash_metrics["純増減"] < cash_metrics["総流出"] * 0.1:
                st.warning("⚠️ 増減が小さく、資金運用の余裕が乏しい可能性があります。")
            else:
                st.success("✅ キャッシュポジションは健全です。")
        else:
            st.info("数値データが見つからないためキャッシュ分析をスキップします。")

    with tab4:
        if df_diff.empty:
            st.info("差異のある列がないためAI示唆はありません。")
        else:
            st.subheader("■ 差異原因示唆 (Gemini 2.5)")
            st.markdown(ai_text)

if __name__ == "__main__":
    main()
