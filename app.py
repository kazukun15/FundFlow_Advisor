import io, re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import google.generativeai as genai

# ─── Secrets から APIキー取得 ────────────────────────────
if "google" not in st.secrets or "api_key" not in st.secrets["google"]:
    st.error('`.streamlit/secrets.toml` に以下を登録してください:\n\n'
             '[google]\n'
             'api_key = "YOUR_GOOGLE_API_KEY"')
    st.stop()
genai.configure(api_key=st.secrets["google"]["api_key"])

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
        tbls = []
        for p in pdf.pages:
            for t in p.extract_tables() or []:
                if len(t) > 1:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    df.columns = make_unique([str(c).strip() for c in df.columns])
                    tbls.append(df)
    if not tbls:
        return pd.DataFrame()
    all_cols = sorted({c for df in tbls for c in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in tbls]
    return pd.concat(aligned, ignore_index=True, sort=False)

def ocr_text(buf: bytes) -> str:
    imgs = convert_from_bytes(buf)
    return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in imgs)

def process_public_fund_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    公金日計特有のヘッダー ('月計処理月' を含む行) を検出し、
    以降をデータとして切り出す。不要なフッター行を削除。
    """
    # 1) ヘッダー行を検出
    mask = raw.apply(lambda row: row.astype(str).str.contains("月計処理月").any(), axis=1)
    if not mask.any():
        return raw  # 見つからなければそのまま返す
    header_idx = mask.idxmax()
    headers = raw.iloc[header_idx].tolist()
    df = raw.iloc[header_idx+1:].copy()
    df.columns = [str(h).strip() for h in headers]

    # 2) フッター（"令和"、"小計"、"ページ"）を含む行を削除
    footer_pattern = re.compile(r"令和|小計|ページ")
    df = df[~df.iloc[:, 0].astype(str).str.contains(footer_pattern)]

    return df.reset_index(drop=True)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = process_public_fund_df(df)   # 公金日計向け前処理を挟む
    # 列名全角・半角スペース除去
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    # 各セルから数字のみ抽出し数値化
    for c in df.columns:
        cleaned = (
            df[c].astype(str)
                 .map(lambda x: re.sub(r"[^\d\-]", "", x))
                 .replace({"": pd.NA})
        )
        df[c] = pd.to_numeric(cleaned, errors="ignore")
    # 完全に空の行・列を削除
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df.reset_index(drop=True)

# （以下、 reconcile・analyze_cf・ai_suggest・fund_advice などは変更なし）
# ─── Streamlit UI ───────────────────────────────────────
def main():
    st.set_page_config(page_title="FundFlow Advisor AI", layout="wide")
    st.title("FundFlow Advisor AI")
    st.markdown("公金日計PDFとその他日報をアップロードし、分析・AI提案を行います。")

    files = st.sidebar.file_uploader(
        "📁 PDFをアップロード（公金日計は必須、その他は任意）",
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
        raw = extract_tables_from_pdf(buf)
        if raw.empty:
            st.sidebar.warning(f"{f.name} 表抽出失敗→OCR表示")
            st.sidebar.text_area(f"OCR({f.name})", ocr_text(buf), height=150)
            continue
        df = normalize_df(raw)
        if pub_df.empty:
            pub_df = df; st.sidebar.success(f"{f.name} を公金日計として設定")
        else:
            others[f.name] = df; st.sidebar.success(f"{f.name} を他日報に追加")

    if pub_df.empty:
        st.warning("公金日計PDFが必要です。")
        return

    # 以降はこれまでと同様に tab表示で突合・分析・AI提案など
    # ...

if __name__ == "__main__":
    main()
