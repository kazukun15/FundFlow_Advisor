import io, os, re
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
    # 列名：全角・半角スペース除去
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    for c in df.columns:
        s = (
            df[c]
            .astype(str)
            .map(lambda x: re.sub(r"[^\d\.\-]", "", x))
            .replace({"": pd.NA})
        )
        df[c] = pd.to_numeric(s, errors="ignore")
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def reconcile_reports(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    base = pub.select_dtypes(include="number").sum().sum()
    rows = []
    for name, df in others.items():
        total = df.select_dtypes(include="number").sum().sum()
        rows.append({
            "レポート": name,
            "公金日計合計": base,
            "他日報合計": total,
            "差異": total - base
        })
    return pd.DataFrame(rows)

def analyze_cash_flow(pub: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    # 「会計前日残高」があれば先頭行で開閉バランスとして抽出
    ob = pub["会計前日残高"].dropna().iloc[0] if "会計前日残高" in pub.columns else None

    num = pub.select_dtypes(include="number")
    col_sums = num.sum().rename("合計").to_frame()

    # 優先列選定
    if "入金" in num.columns and "出金" in num.columns:
        inflow = num["入金"].sum()
        outflow = num["出金"].sum()
    elif "収入" in num.columns and "支出" in num.columns:
        inflow = num["収入"].sum()
        outflow = num["支出"].sum()
    elif "月計収支" in num.columns:
        # 月計収支 = 収入-支出 と仮定
        inflow = num["月計収支"][num["月計収支"] > 0].sum()
        outflow = -num["月計収支"][num["月計収支"] < 0].sum()
    else:
        inflow = num[num>0].sum().sum()
        outflow = -num[num<0].sum().sum()

    net = inflow - outflow
    metrics = {"始値(前日残高)": ob, "総流入": inflow, "総流出": outflow, "純増減": net}
    return metrics, col_sums

def generate_ai_suggestions(df_diff: pd.DataFrame) -> str:
    table = df_diff[df_diff["差異"] != 0].to_string(index=False) or "（全レポート差異なし）"
    prompt = (
        "以下の差異について、原因を箇条書きで示してください。\n\n"
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
        "- サイドバーでPDFをアップロード\n"
        "- 上部タブで「プレビュー」「差異サマリー」「分析」「AI示唆」"
    )

    uploaded = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF", type=None, accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.info("ここでPDFをアップロードしてください。")
        return

    pub_df, others = pd.DataFrame(), {}
    for f in uploaded:
        if not f.name.lower().endswith(".pdf"):
            st.sidebar.error(f"{f.name} はPDFではありません")
            continue
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{f.name} 抽出失敗。OCRを表示")
            st.sidebar.text_area(f"OCR({f.name})", fallback_ocr_pdf(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df; st.sidebar.success(f"{f.name} を基準に設定")
        else:
            others[f.name] = df; st.sidebar.success(f"{f.name} を他日報に追加")

    if pub_df.empty or not others:
        st.warning("基準と他日報、両方のPDFが必要です。")
        return

    df_diff = reconcile_reports(pub_df, others)
    cash_metrics, col_sums = analyze_cash_flow(pub_df)
    ai_text = generate_ai_suggestions(df_diff)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 プレビュー", "📊 差異サマリー", "💡 分析", "🤖 AI示唆"]
    )

    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.write("**列名一覧(正規化後)**", list(pub_df.columns))
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報プレビュー：{name}")
            st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("■ 差異サマリー")
        st.dataframe(df_diff, use_container_width=True)

    with tab3:
        st.subheader("■ キャッシュフロー／列合計分析")
        # 前日残高
        if cash_metrics["始値(前日残高)"] is not None:
            st.metric("前日残高", f"{int(cash_metrics['始値(前日残高)']):,}")
        # 流入・流出・純増減
        c1, c2, c3 = st.columns(3)
        c1.metric("総流入", f"{int(cash_metrics['総流入']):,}")
        c2.metric("総流出", f"{int(cash_metrics['総流出']):,}")
        c3.metric("純増減", f"{int(cash_metrics['純増減']):,}")
        st.subheader("◾ 列ごとの合計値")
        st.dataframe(col_sums, use_container_width=True)
        # リスク判定
        if cash_metrics["純増減"] < 0:
            st.error("⚠️ 資金ショートのリスクがあります。")
        elif cash_metrics["純増減"] < cash_metrics["総流出"] * 0.1:
            st.warning("⚠️ 資金余裕が乏しい可能性があります。")
        else:
            st.success("✅ キャッシュポジションは健全です。")

    with tab4:
        st.subheader("■ 差異原因示唆 (Gemini 2.5)")
        st.markdown(ai_text)

if __name__ == "__main__":
    main()
