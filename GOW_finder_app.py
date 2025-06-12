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

def label_company(company_name):
    prompt = f"""
You are an expert marine engineering consultant. Your task is to label companies based on their name and your knowledge of the marine and offshore industries.

For the company: "{company_name}"

Return a JSON object with the following keys:
{{
    "industry": "The company's primary industry sector (e.g., 'Offshore Wind', 'Marine Transportation', 'Port Operations', 'Marine Engineering', 'Energy', 'Construction', etc.)",
    "company_type": "The company's role in the industry (e.g., 'Developer', 'OEM', 'Engineering Consultant', 'Port Authority', 'Contractor', 'Operator', 'Manufacturer', etc.)",
    "business_model": "The company's business model (e.g., 'Project Developer', 'Service Provider', 'Equipment Manufacturer', 'Consultancy', etc.)",
    "company_size": "Estimated company size (e.g., '1-10', '11-50', '51-200', '201+', 'Unknown'). If you have no information, use 'Unknown'.",
    "company_age": "Estimated company age or maturity (e.g., 'Startup', 'Young', 'Established', 'Unknown').",
    "potential_partner": "true if this company is likely a good small/young partner for a small consultancy (<5 people), otherwise false.",
    "partner_reasoning": "A short explanation for your potential_partner assessment.",
    "relevant_offerings": ["List of relevant services from: {', '.join(offerings_list)} that could be of use to this company from a third party"],
    "reasoning": "A short explanation of why these offerings are relevant to this company."
}}

Guidelines:
- Only include offerings that would be genuinely relevant to this type of company
- If uncertain about any field, use 'Unknown' rather than guessing
- If the company seems small, new, or a potential partner, set potential_partner to true and explain why
- Ensure the response is valid JSON
- Respond with JSON only
"""
    try:
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a precise and accurate marine engineering industry labeler. Always respond with valid JSON."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        # Validate the response structure
        required_keys = [
            "industry", "company_type", "business_model", "company_size", "company_age",
            "potential_partner", "partner_reasoning", "relevant_offerings", "reasoning"
        ]
        if not all(key in result for key in required_keys):
            raise ValueError("Missing required keys in response")
        if not isinstance(result["relevant_offerings"], list):
            result["relevant_offerings"] = []
        result["relevant_offerings"] = [
            offering for offering in result["relevant_offerings"]
            if offering in offerings_list
        ]
        return result
    except Exception as e:
        st.error(f"Error labeling {company_name}: {str(e)}")
        return {
            "industry": "Error",
            "company_type": "Error",
            "business_model": "Error",
            "company_size": "Error",
            "company_age": "Error",
            "potential_partner": False,
            "partner_reasoning": str(e),
            "relevant_offerings": [],
            "reasoning": str(e)
        }

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

    # --- GPT Labeling ---
    test_mode = st.toggle("🧪 Test Mode (Process only first 5 companies)", value=True)
    
    if st.button("🔍 Label Companies with GPT"):
        st.info("Calling GPT to label each unique company...")
        progress = st.progress(0)
        result_map = {}
        companies = df["Company"].unique()
        
        if test_mode:
            companies = companies[:5]
            st.warning("🧪 Test Mode: Processing only first 5 companies")

        # Custom loading spinner in main area
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div style='display: flex; align-items: center; gap: 0.5em;'>
                <span style='font-size:2em;'>⏳</span>
                <span style='font-size:1.2em;'>Labeling companies, please wait...</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, company in enumerate(companies):
            with st.spinner(f"Labeling {company}..."):
                result_map[company] = label_company(company)
                progress.progress((i + 1) / len(companies))
                time.sleep(1.1)

        loading_placeholder.empty()  # Remove the custom spinner when done

        # Attach results to DataFrame
        df["Industry"] = df["Company"].map(lambda x: result_map.get(x, {}).get("industry", ""))
        df["Company Type"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_type", ""))
        df["Business Model"] = df["Company"].map(lambda x: result_map.get(x, {}).get("business_model", ""))
        df["Company Size"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_size", ""))
        df["Company Age"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_age", ""))
        df["Potential Partner"] = df["Company"].map(lambda x: result_map.get(x, {}).get("potential_partner", False))
        df["Partner Reasoning"] = df["Company"].map(lambda x: result_map.get(x, {}).get("partner_reasoning", ""))
        df["Relevant Offerings"] = df["Company"].map(
            lambda x: ", ".join(result_map.get(x, {}).get("relevant_offerings", []))
        )
        df["Reasoning"] = df["Company"].map(lambda x: result_map.get(x, {}).get("reasoning", ""))

        st.success("✅ Labeling complete.")
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
