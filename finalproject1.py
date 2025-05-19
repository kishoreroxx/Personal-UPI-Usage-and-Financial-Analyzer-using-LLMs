import os
import re
import pandas as pd
import pdfplumber
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- HELPER FUNCTIONS ----------

def extract_transactions_from_pdf(pdf_bytes):
    """
    Extract transaction data from a UPI statement PDF file.
    """
    data = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            transactions = parse_text_to_transactions(text)
            data.extend(transactions)

    df = pd.DataFrame(data, columns=['Date', 'Time', 'Amount', 'Receiver', 'Description'])
    return df

def parse_text_to_transactions(text):
    """
    Parse text from a PDF page into structured transaction data.
    """
    transactions = []
    lines = text.split('\n')
    for line in lines:
        fields = line.split()
        if validate_transaction(fields):
            try:
                date = fields[0]
                time = fields[1]
                amount = float(fields[2].replace(',', ''))
                receiver = fields[3]
                description = ' '.join(fields[4:])
                transactions.append([date, time, amount, receiver, description])
            except:
                continue
    return transactions

def validate_transaction(fields):
    """
    Improved validation for transaction lines using regex.
    """
    return len(fields) >= 5 and re.match(r'^\d+(\.\d{1,2})?$', fields[2].replace(',', ''))

def categorize_transaction(description):
    """
    Categorize transactions based on keywords.
    """
    desc = description.lower()
    if any(x in desc for x in ['zomato', 'swiggy', 'food']):
        return 'Food'
    elif any(x in desc for x in ['uber', 'ola', 'cab']):
        return 'Transport'
    elif any(x in desc for x in ['amazon', 'flipkart', 'shopping']):
        return 'Shopping'
    elif any(x in desc for x in ['electricity', 'bill', 'recharge']):
        return 'Utilities'
    elif any(x in desc for x in ['rent', 'apartment', 'flat']):
        return 'Housing'
    else:
        return 'Others'

def generate_financial_recommendations(data):
    """
    Generate LLM-based financial recommendations using Gemini API.
    """
    if os.getenv("GEMINI_API_KEY") is None:
        return "Gemini API key not set. Please set GEMINI_API_KEY in your environment."

    summary = data.groupby('Category')['Amount'].sum().to_dict()
    wasteful = data[data['Amount'] <= 50].to_dict(orient='records')

    prompt = f"""
    Analyze the following transaction category summary:
    {summary}

    Also, here are some transactions under ₹50 that may be wasteful:
    {wasteful}

    Provide personalized advice to reduce unnecessary expenses and improve budgeting.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating recommendations: {e}"

# ---------- GRADIO INTERFACE ----------

def analyze_pdf(pdf_file):
    try:
        if pdf_file is None:
            return "Please upload a PDF file.", None, None, None, None, None

        pdf_bytes = pdf_file

        df = extract_transactions_from_pdf(pdf_bytes)

        if df.empty:
            return "No transactions could be extracted. Please check your PDF format.", None, None, None, None, None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        df['Category'] = df['Description'].apply(categorize_transaction)

        recommendations = generate_financial_recommendations(df)

        transactions_html = df.to_html(index=False)
        monthly_spending = df.groupby('Month')['Amount'].sum().reset_index()
        category_spending = df.groupby('Category')['Amount'].sum().reset_index()
        wasteful_df = df[df['Amount'] <= 50]
        wasteful_html = wasteful_df.to_html(index=False) if not wasteful_df.empty else "No low-value transactions detected."

        # Save cleaned transactions as CSV
        csv_path = os.path.join(tempfile.gettempdir(), "cleaned_transactions.csv")
        df.to_csv(csv_path, index=False)

        return transactions_html, monthly_spending, category_spending, wasteful_html, recommendations, csv_path

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None, None, None, None, None

# ---------- INTERFACE ----------

interface = gr.Interface(
    fn=analyze_pdf,
    inputs=gr.File(label="Upload your UPI Statement (PDF)", type="binary"),
    outputs=[
        gr.HTML(label="All Transactions"),
        gr.Dataframe(label="Monthly Spending Trend"),
        gr.Dataframe(label="Category-wise Spending"),
        gr.HTML(label="Wasteful Transactions (≤ ₹50)"),
        gr.Textbox(label="LLM Financial Recommendations"),
        gr.File(label="Download Cleaned Transactions CSV")
    ],
    title="📊 Personal UPI Financial Analyzer",
    description="Upload your UPI statement PDF to analyze your spending and get financial recommendations."
)

# ---------- LAUNCH ----------
interface.launch(share=True)