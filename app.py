import io, re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── Streamlit Secrets から google.api_key を取得 ───────────────────────────
if "google" not in st.secrets or "api_key" not in st.secrets["google"]:
    st.error('`.streamlit/secrets.toml` に以下を登録してください:\n\n'
             '[google]\n'
             'api_key = "YOUR_GOOGLE_API_KEY"')
    st.stop()
genai.configure(api_key=st.secrets["google"]["api_key"])

# ─── ユーティリティ関数 ─────────────────────────────────────
def make_unique(cols):
    seen = {}; out = []
    for c in cols:
        cnt = seen.get(c, 0)
        name = f"{c}.{cnt}" if cnt else c
        seen[c] = cnt + 1
        out.append(name)
    return out

def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        tbls = []
        for p in pdf.pages:
            for t in p.extract_tables() or []:
                if len(t) > 1:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    df.columns = make_unique([str(c).strip() for c in df.columns])
                    tbls.append(df)
    if not tbls:
        return pd.DataFrame()
    cols = sorted({c for df in tbls for c in df.columns})
    aligned = [df.reindex(columns=cols) for df in tbls]
    return pd.concat(aligned, ignore_index=True, sort=False)

def ocr_text(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    for c in df.columns:
        s = (
            df[c].astype(str)
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
        rows.append({"レポート": name, "公金日計合計": base, "他日報合計": tot, "差異": tot - base})
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
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        resp  = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        if hasattr(resp, "parts"):
            return "".join(p.text for p in resp.parts).strip()
        st.warning("AI応答形式が予期せぬものでした。")
        return "AIから有効な提案を取得できませんでした。"
    except Exception as e:
        st.error(f"AI提案生成エラー: {e}")
        return "AI提案の生成中にエラーが発生しました。"

def fund_advice(pub: pd.DataFrame) -> (dict, str):
    # 「基金」を含む列を検出
    fund_cols = [c for c in pub.columns if "基金" in c]
    if not fund_cols:
        return {}, "基金データが見つかりませんでした。"
    # 金額合計
    fund_sums = {c: pub[c].sum() for c in fund_cols if pd.api.types.is_numeric_dtype(pub[c])}
    # AIプロンプト作成
    table = "\n".join(f"{c}: {v:,}" for c, v in fund_sums.items())
    prompt = (
        "以下は基金の残高情報です。\n"
        f"{table}\n"
        "これらを一般会計へ繰り入れる、または償還に利用する際の注意点とおすすめの方法を、"
        "箇条書きで3～5点挙げてください。"
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        resp  = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            advice = resp.text.strip()
        elif hasattr(resp, "parts"):
            advice = "".join(p.text for p in resp.parts).strip()
        else:
            advice = "AIから基金アドバイスを取得できませんでした。"
    except Exception as e:
        st.error(f"基金アドバイス生成エラー: {e}")
        advice = "基金アドバイスの生成中にエラーが発生しました。"
    return fund_sums, advice

# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor AI", layout="wide")
    st.title("FundFlow Advisor AI")
    st.markdown("PDFをアップロードし、突合・キャッシュ分析・AI提案・基金アドバイスを行います。")

    files = st.sidebar.file_uploader(
        "📁 公金日計PDF と 他日報PDF（複数可）",
        type=None, accept_multiple_files=True
    )
    if not files:
        st.sidebar.info("ここにPDFをアップロードしてください。")
        return

    pub_df = pd.DataFrame()
    others = {}
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            st.sidebar.error(f"{f.name} はPDFではありません"); continue
        buf = f.read()
        df  = extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{f.name} 表抽出失敗→OCR表示")
            st.sidebar.text_area(f"OCR({f.name})", ocr_text(buf), height=150)
            continue
        df = normalize_df(df)
        if pub_df.empty:
            pub_df = df; st.sidebar.success(f"{f.name} を公金日計に設定")
        else:
            others[f.name] = df; st.sidebar.success(f"{f.name} を他日報に追加")

    if pub_df.empty:
        st.warning("公金日計PDFが必要です。")
        return

    df_diff    = reconcile(pub_df, others) if others else pd.DataFrame()
    cf_metrics, cf_sums = analyze_cf(pub_df)
    diff_ai    = ai_suggest(df_diff)     if not df_diff.empty else "他日報がないため突合AIはありません。"
    fund_sums, fund_ai = fund_advice(pub_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 プレビュー","📊 差異サマリー","💡 分析","🤖 AI突合提案","🏦 基金アドバイス"]
    )

    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df, use_container_width=True)
        if others:
            st.subheader("■ 他日報プレビュー")
            for name, df in others.items():
                st.markdown(f"**{name}**")
                st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("■ 差異サマリー")
        if df_diff.empty:
            st.info("他日報がないため差異サマリーはありません。")
        else:
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
        st.subheader("■ AIによる突合原因提案")
        st.markdown(diff_ai)

    with tab5:
        st.subheader("■ 基金アドバイス")
        if fund_sums:
            st.table(pd.DataFrame.from_dict(fund_sums, orient="index", columns=["残高"]).assign(残高=lambda df: df["残高"].map("{:,}".format)))
        else:
            st.info("基金データが見つかりませんでした。")
        st.markdown(fund_ai)

if __name__ == "__main__":
    main()
