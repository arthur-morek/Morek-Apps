import streamlit as st
import pandas as pd
import pdfplumber
import openai
import json
import time

# --- Settings ---
st.title("📄 GOW 2025 Delegate Extractor + GPT Company Classifier")

offerings_list = [
    "naval architecture support",
    "temporary mooring design",
    "detailed mooring design",
    "pre-FEED mooring and dynamic cable design",
    "seafastening",
    "marine operations planning"
]

client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

def classify_company(company_name):
    prompt = f"""
You are helping a marine engineering consultancy understand prospective clients.

Classify this company: "{company_name}"

Return JSON with keys:
- main_sector: the company's core focus or industry
- company_type: e.g. developer, OEM, engineering consultant, port, contractor, etc.
- relevant_offerings: a list of the offerings below that could be relevant

Offerings: {offerings_list}

Respond in JSON only.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"main_sector": "", "company_type": "", "relevant_offerings": [], "error": str(e)}

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload Delegate List PDF", type=["pdf"])

if uploaded_file:
    # Extract table by layout
    rows = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            grouped = {}
            y_tol = 3

            for word in words:
                y = round(float(word["top"]) / y_tol) * y_tol
                grouped.setdefault(y, []).append(word)

            for y in sorted(grouped.keys()):
                line = sorted(grouped[y], key=lambda w: w["x0"])
                names, jobs, comps = [], [], []
                for word in line:
                    x = word["x0"]
                    if x < 150:
                        names.append(word["text"])
                    elif 150 <= x < 350:
                        jobs.append(word["text"])
                    else:
                        comps.append(word["text"])
                if names and jobs and comps:
                    first = names[0]
                    last = " ".join(names[1:]) if len(names) > 1 else ""
                    job = " ".join(jobs)
                    comp = " ".join(comps)
                    rows.append([first, last, job, comp])

    df = pd.DataFrame(rows, columns=["First Name", "Last Name", "Job Title", "Company"])
    st.success(f"✅ Extracted {len(df)} delegates.")
    st.dataframe(df)

    # --- GPT Classification ---
    if st.button("🔍 Classify Companies with GPT"):
        st.info("Calling GPT to classify each unique company...")
        progress = st.progress(0)
        result_map = {}
        companies = df["Company"].unique()

        for i, company in enumerate(companies):
            result_map[company] = classify_company(company)
            progress.progress((i + 1) / len(companies))
            time.sleep(1.1)  # avoid hitting rate limits

        # Attach results to DataFrame
        df["Main Sector"] = df["Company"].map(lambda x: result_map[x].get("main_sector", ""))
        df["Company Type"] = df["Company"].map(lambda x: result_map[x].get("company_type", ""))
        df["Relevant Offerings"] = df["Company"].map(
            lambda x: ", ".join(result_map[x].get("relevant_offerings", []))
        )

        st.success("✅ Classification complete.")
        st.dataframe(df)

    # --- Search ---
    search_term = st.text_input("🔍 Search by name, job title, company, or offering:")
    if search_term:
        # Create a mask for each column and combine them
        mask = pd.Series(False, index=df.index)
        for column in df.columns:
            mask |= df[column].astype(str).str.contains(search_term, case=False, na=False)
        
        filtered = df[mask]
        st.write(f"🔎 Found {len(filtered)} result(s):")
        st.dataframe(filtered)

    # --- Download ---
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="gow2025_delegates_classified.csv")

else:
    st.info("Upload a GOW 2025 delegate list PDF to begin.")
