import io, os, re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── 設定 ─────────────────────────────────────────────
st.set_page_config(page_title="FundFlow Advisor AI", layout="wide")
API_SECRET = "GOOGLE_API_KEY"
api_key = st.secrets.get(API_SECRET) or os.getenv(API_SECRET)
if not api_key:
    st.error(f"❌ `{API_SECRET}` に APIキーを設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# ─── ユーティリティ関数 ─────────────────────────────────
def make_unique(cols):
    seen = {}; out = []
    for c in cols:
        cnt = seen.get(c,0)
        name = f"{c}.{cnt}" if cnt else c
        seen[c] = cnt+1; out.append(name)
    return out

def extract_tables_from_pdf(buf: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(buf)) as pdf:
        tbls=[]
        for p in pdf.pages:
            for t in p.extract_tables() or []:
                if len(t)>1:
                    df=pd.DataFrame(t[1:],columns=t[0])
                    df.columns=make_unique([str(c).strip() for c in df.columns])
                    tbls.append(df)
    if not tbls: return pd.DataFrame()
    cols=sorted({c for df in tbls for c in df.columns})
    aligned=[df.reindex(columns=cols) for df in tbls]
    return pd.concat(aligned,ignore_index=True,sort=False)

def ocr_text(buf: bytes)->str:
    imgs=convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(i,lang="jpn") for i in imgs)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df.columns=[re.sub(r"\s+","",str(c)) for c in df.columns]
    for c in df.columns:
        s=(df[c].astype(str)
             .map(lambda x: re.sub(r"[^\d\.\-]","",x))
             .replace({"":pd.NA}))
        df[c]=pd.to_numeric(s,errors="ignore")
    df.dropna(axis=0,how="all",inplace=True)
    df.dropna(axis=1,how="all",inplace=True)
    return df

def reconcile(pub:pd.DataFrame, others:dict)->pd.DataFrame:
    base=pub.select_dtypes("number").sum().sum()
    rows=[]
    for name,df in others.items():
        tot=df.select_dtypes("number").sum().sum()
        rows.append({"レポート":name,"公金日計合計":base,"他日報合計":tot,"差異":tot-base})
    return pd.DataFrame(rows)

def analyze_cf(pub:pd.DataFrame):
    ob=pub.get("会計前日残高",pd.Series(dtype=float)).dropna()
    ob=ob.iloc[0] if not ob.empty else None
    nums=pub.select_dtypes("number")
    sums=nums.sum().rename("合計").to_frame()
    if "入金" in nums and "出金" in nums:
        infl=nums["入金"].sum(); outf=nums["出金"].sum()
    elif "収入" in nums and "支出" in nums:
        infl=nums["収入"].sum(); outf=nums["支出"].sum()
    else:
        infl=nums[nums>0].sum().sum(); outf=-nums[nums<0].sum().sum()
    net=infl-outf
    metrics={"前日残高":ob,"総流入":infl,"総流出":outf,"純増減":net}
    return metrics,sums

def ai_suggest(df_diff:pd.DataFrame)->str:
    diff=df_diff[df_diff["差異"]!=0]
    txt=diff.to_string(index=False) if not diff.empty else "（差異なし）"
    prompt=(
        "以下の差異について、原因を箇条書きで示してください。\n\n"+txt
    )
    # Gemini呼び出し
    resp=genai.chat.completions.create(
        model="gemini-2.5",
        prompt=[{"author":"user","content":prompt}],
        temperature=0.7
    )
    return resp.candidates[0].message.content

# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.title("FundFlow Advisor AI 💡")
    st.markdown("PDFをアップロードして突合・分析・AI提案まで一気通貫。")

    files=st.sidebar.file_uploader("📁 PDFを複数選択",type=None,accept_multiple_files=True)
    if not files:
        st.sidebar.info("ここにPDFをアップロードしてください。")
        return

    pub_df=pd.DataFrame(); others={}
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            st.sidebar.error(f"{f.name} はPDFではありません"); continue
        buf=f.read(); df=extract_tables_from_pdf(buf)
        if df.empty:
            st.sidebar.warning(f"{f.name} 表検出失敗→OCR表示"); st.sidebar.text_area(f"OCR({f.name})",ocr_text(buf),height=150)
            continue
        df=normalize_df(df)
        if pub_df.empty:
            pub_df=df; st.sidebar.success(f"基準に {f.name}")
        else:
            others[f.name]=df; st.sidebar.success(f"他日報に {f.name}")

    if pub_df.empty or not others:
        st.warning("基準PDFと他日報PDFをそれぞれアップロードしてください。"); return

    df_diff=reconcile(pub_df,others)
    cf_metrics, cf_sums=analyze_cf(pub_df)
    ai_text=ai_suggest(df_diff)

    tab1,tab2,tab3,tab4=st.tabs(["🔍 プレビュー","📊 差異サマリー","💡 分析","🤖 AI提案"])
    with tab1:
        st.subheader("■ 公金日計プレビュー")
        st.dataframe(pub_df,use_container_width=True)
        for name,df in others.items():
            st.subheader(f"■ 他日報：{name}")
            st.dataframe(df,use_container_width=True)
    with tab2:
        st.subheader("■ 差異サマリー")
        st.dataframe(df_diff,use_container_width=True)
    with tab3:
        st.subheader("■ キャッシュフロー分析")
        if cf_metrics["前日残高"] is not None:
            st.metric("前日残高",f"{int(cf_metrics['前日残高']):,}")
        c1,c2,c3=st.columns(3)
        c1.metric("総流入",f"{int(cf_metrics['総流入']):,}")
        c2.metric("総流出",f"{int(cf_metrics['総流出']):,}")
        c3.metric("純増減",f"{int(cf_metrics['純増減']):,}")
        st.subheader("▪ 列合計デバッグ")
        st.dataframe(cf_sums,use_container_width=True)
        if cf_metrics["純増減"]<0:
            st.error("⚠️ 資金ショートリスク")
        elif cf_metrics["純増減"]<cf_metrics["総流出"]*0.1:
            st.warning("⚠️ 余裕乏しめ")
        else:
            st.success("✅ 健全")
    with tab4:
        st.subheader("■ AIによる原因提案")
        st.markdown(ai_text)

if __name__=="__main__":
    main()
