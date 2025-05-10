import io
import re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── Streamlit Secrets からのみ取得 ─────────────────────────
API_SECRET_KEY = "GOOGLE_API_KEY"
if API_SECRET_KEY not in st.secrets:
    st.error(f"❌ Streamlit Secrets に `{API_SECRET_KEY}` を設定してください。")
    st.stop()
genai.configure(api_key=st.secrets[API_SECRET_KEY])

# ─── ユーティリティ関数 ─────────────────────────────────────
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
                    df.columns = make_unique([str(c).strip() for c in df.columns])
                    tables.append(df)
    if not tables:
        return pd.DataFrame()
    all_cols = sorted({c for df in tables for c in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in tables]
    return pd.concat(aligned, ignore_index=True, sort=False)

def ocr_text(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

def reconcile(pub: pd.DataFrame, others: dict) -> pd.DataFrame:
    base = pub.select_dtypes("number").sum().sum()
    rows = []
    for name, df in others.items():
        tot = df.select_dtypes("number").sum().sum()
        rows.append({
            "レポート": name,
            "公金日計合計": base,
            "他日報合計": tot,
            "差異": tot - base
        })
    return pd.DataFrame(rows)

def analyze_cf(pub: pd.DataFrame):
    ob_series = pub.get("会計前日残高", pd.Series(dtype=float)).dropna()
    ob = ob_series.iloc[0] if not ob_series.empty else None
    nums = pub.select_dtypes("number")
    sums = nums.sum().rename("合計").to_frame()
    if {"入金", "出金"}.issubset(nums.columns):
        infl, outf = nums["入金"].sum(), nums["出金"].sum()
    elif {"収入", "支出"}.issubset(nums.columns):
        infl, outf = nums["収入"].sum(), nums["支出"].sum()
    else:
        infl = nums[nums > 0].sum().sum()
        outf = -nums[nums < 0].sum().sum()
    net = infl - outf
    return {"前日残高": ob, "総流入": infl, "総流出": outf, "純増減": net}, sums

def ai_suggest(df_diff: pd.DataFrame) -> str:
    diff = df_diff[df_diff["差異"] != 0]
    txt = diff.to_string(index=False) if not diff.empty else "（差異なし）"
    prompt = f"以下の差異について、原因を箇条書きで示してください。\n\n{txt}"

    try:
        # 指定されたGeminiモデルを使用
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        response = model.generate_content(prompt)
        # レスポンスからテキストを取り出し
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "parts") and response.parts:
            return "".join(p.text for p in response.parts).strip()
        # フォールバック
        st.warning("AIの応答形式が予期せぬものでした。")
        return "AIから有効な提案を取得できませんでした。"
    except Exception as e:
        st.error(f"AI提案生成エラー: {e}")
        return "AI提案の生成中にエラーが発生しました。"

# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor AI", layout="wide")
    st.title("FundFlow Advisor AI")
    st.markdown("PDFをアップロードして突合・分析・AI提案まで一気通貫。")

    files = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF", type=None, accept_multiple_files=True
    )
    if not files:
        st.sidebar.info("ここにPDFをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            st.sidebar.error(f"{f.name} はPDFではありません")
            continue
        buf = f.read()
        df = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{f.name} 表抽出失敗→OCR表示")
            st.sidebar.text_area(f"OCR({f.name})", ocr_text(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df
            st.sidebar.success(f"{f.name} を基準に設定")
        else:
            others[f.name] = df
            st.sidebar.success(f"{f.name} を他日報に追加")

    if pub_df.empty or not others:
        st.warning("基準PDFと他日報PDFをそれぞれアップロードしてください。")
        return

    df_diff = reconcile(pub_df, others)
    cf_metrics, cf_sums = analyze_cf(pub_df)
    ai_text = ai_suggest(df_diff)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 プレビュー", "📊 差異サマリー", "💡 分析", "🤖 AI提案"]
    )
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df, use_container_width=True)
        for name, df in others.items():
            st.subheader(f"■ 他日報：{name}")
            st.dataframe(df, use_container_width=True)
    with tab2:
        st.subheader("■ 差異サマリー")
        st.dataframe(df_diff, use_container_width=True)
    with tab3:
        st.subheader("■ キャッシュフロー分析")
        if cf_metrics["前日残高"] is not None:
            st.metric("前日残高", f"{int(cf_metrics['前日残高']):,}")
        c1, c2, c3 = st.columns(3)
        c1.metric("総流入", f"{int(cf_metrics['総流入']):,}")
        c2.metric("総流出", f"{int(cf_metrics['総流出']):,}")
        c3.metric("純増減", f"{int(cf_metrics['純増減']):,}")
        st.subheader("▪ 列合計デバッグ")
        st.dataframe(cf_sums, use_container_width=True)
        if cf_metrics["純増減"] < 0:
            st.error("⚠️ 資金ショートリスク")
        elif cf_metrics["純増減"] < cf_metrics["総流出"] * 0.1:
            st.warning("⚠️ 資金余裕が乏しい可能性があります")
        else:
            st.success("✅ キャッシュポジションは健全です")
    with tab4:
        st.subheader("■ AIによる原因提案")
        st.markdown(ai_text)

if __name__ == "__main__":
    main()
