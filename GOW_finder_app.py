import streamlit as st
import pandas as pd
import pdfplumber

st.title("📄 GOW 2025 Delegate Extractor (Layout-Aware)")

uploaded_file = st.file_uploader("📎 Upload Delegate List PDF", type=["pdf"])

if uploaded_file:
    # Load PDF and extract rows based on x0 column position
    position_based_rows = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            rows = {}
            y_tolerance = 3

            for word in words:
                y = round(float(word["top"]) / y_tolerance) * y_tolerance
                rows.setdefault(y, []).append(word)

            for y_key in sorted(rows.keys()):
                row_words = sorted(rows[y_key], key=lambda w: w["x0"])
                name_parts, job_parts, company_parts = [], [], []

                for word in row_words:
                    x = word["x0"]
                    if x < 150:
                        name_parts.append(word["text"])
                    elif 150 <= x < 350:
                        job_parts.append(word["text"])
                    else:
                        company_parts.append(word["text"])

                if len(name_parts) >= 1 and job_parts and company_parts:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                    job_title = " ".join(job_parts)
                    company = " ".join(company_parts)
                    position_based_rows.append([first_name, last_name, job_title, company])

    # Create DataFrame
    df = pd.DataFrame(position_based_rows, columns=["First Name", "Last Name", "Job Title", "Company"])

    st.success(f"✅ Extracted {len(df)} delegates.")
    st.dataframe(df)

    search_term = st.text_input("🔍 Search by name, job title, or company:")

    if search_term:
        results = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        st.write(f"🔎 Found {len(results)} result(s):")
        st.dataframe(results)

    st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="gow2025_delegates.csv")

else:
    st.info("Upload a GOW 2025 delegate list PDF to begin.")


