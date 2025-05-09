# Write the updated requirements.txt
requirements_txt = """\
streamlit
pdfplumber
pytesseract
pdf2image
pillow
pandas
openai
"""

with open('/mnt/data/requirements.txt', 'w') as f:
    f.write(requirements_txt)

# Write the updated app code with Gemini 2.5 model
app_code = """import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import openai

# --- 設定 ---
openai.api_key = st.secrets["openai"]["api_key"]

# --- PDF抽出関数 ---
def extract_tables_from_pdf(file_bytes):
    \"\"\"PDFからテーブルを抽出してDataFrameとして返す\"\"\"
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    dfs = []
    for table in tables:
        if table and len(table) > 1:
            df = pd.DataFrame(table[1:], columns=table[0])
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def fallback_ocr_pdf(file_bytes):
    \"\"\"OCRでPDFを画像化し、テキストを抽出\"\"\"
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='jpn')
    return text

# --- 正規化関数 ---
def normalize_df(df):
    \"\"\"数値カラムを正しく認識させるクリーニング処理\"\"\"
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

# --- 突合関数 ---
def reconcile_reports(pub_df, other_dfs):
    \"\"\"公金日計と他日報を比較し、差異をまとめる\"\"\"
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

# --- AI示唆生成関数 ---
def generate_ai_suggestions(suggestions):
    \"\"\"差異情報をもとに原因示唆をAIに生成させる\"\"\"
    prompt = (
        "以下の日報突合結果について、"
        "なぜ差異が発生したかの可能性を箇条書きで示唆してください。\\n"
        f"{suggestions}"
    )
    response = openai.ChatCompletion.create(
        model='gemini-2.5',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
def main():
    st.title("🗒 日報作成支援アプリ")
    st.markdown("公金日計PDFと各部署の日報PDFをアップロードし、突合やAI示唆を行います。")

    pub_file = st.file_uploader("📑 公金日計PDFをアップロード", type=['pdf'])
    other_files = st.file_uploader("📑 他部署日報PDFをアップロード（複数可）", type=['pdf'], accept_multiple_files=True)

    if pub_file and other_files:
        # 公金日計PDF解析
        bytes_pub = pub_file.read()
        try:
            pub_df = extract_tables_from_pdf(bytes_pub)
            st.success("✅ 公金日計PDFのテーブルを抽出しました。")
        except Exception:
            st.warning("⚠️ テーブル抽出に失敗したためOCRを実行します。")
            text = fallback_ocr_pdf(bytes_pub)
            st.text_area("OCRテキスト（公金日計）", text, height=200)
            pub_df = pd.DataFrame()  # OCR結果は手動確認用

        pub_df = normalize_df(pub_df)

        # 他日報PDF解析
        other_dfs = {}
        for f in other_files:
            bytes_f = f.read()
            try:
                df = extract_tables_from_pdf(bytes_f)
                st.success(f"✅ {f.name} のテーブルを抽出しました。")
            except Exception:
                st.warning(f"⚠️ {f.name} はテーブル抽出に失敗。OCRを表示します。")
                text = fallback_ocr_pdf(bytes_f)
                st.text_area(f"OCRテキスト（{f.name}）", text, height=200)
                df = pd.DataFrame()
            other_dfs[f.name] = normalize_df(df)

        # 差異計算と表示
        suggestions = reconcile_reports(pub_df, other_dfs)
        if suggestions:
            st.subheader("▶ 差異サマリ")
            st.table(pd.DataFrame(suggestions))
            st.subheader("▶ AIからの原因示唆")
            ai_text = generate_ai_suggestions(suggestions)
            st.write(ai_text)
        else:
            st.success("🎉 日報間に差異は検出されませんでした。")
    else:
        st.info("まずはすべてのPDFをアップロードしてください。")

if __name__ == "__main__":
    main()
"""

with open('/mnt/data/app_updated.py', 'w') as f:
    f.write(app_code)

# Display links for download
print("[Download requirements.txt](sandbox:/mnt/data/requirements.txt)")
print("[Download updated app code](sandbox:/mnt/data/app_updated.py)")
