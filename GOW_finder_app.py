import streamlit as st
import pandas as pd
import pdfplumber
import openai
import json
import time
import altair as alt
import os

# --- Settings ---
st.title("📄 GOW 2025 Delegate Extractor + GPT Company Classifier")

offerings_list = [
    "naval architecture support",
    "temporary mooring design",
    "detailed mooring design",
    "pre-FEED mooring and dynamic cable design",
    "seafastening",##
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
    "known_partners": ["List of known or likely partner companies, if any are public or can be inferred. If unknown, return an empty list."],
    "similar_to_morek": ["List of companies that are similar to Morek Engineering, if any. If unknown, return an empty list."],
    "relevant_offerings": ["List of relevant services from: {', '.join(offerings_list)} that could be of use to this company from a third party"],
    "reasoning": "A short explanation of why these offerings are relevant to this company."
}}

Guidelines:
- Only include offerings that would be genuinely relevant to this type of company
- If uncertain about any field, use 'Unknown' rather than guessing
- If the company seems small, new, or a potential partner, set potential_partner to true and explain why
- If you know or can infer any partners, list them in known_partners
- If you know or can infer any companies similar to Morek Engineering, list them in similar_to_morek
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
            "potential_partner", "partner_reasoning", "known_partners", "similar_to_morek",
            "relevant_offerings", "reasoning"
        ]
        if not all(key in result for key in required_keys):
            raise ValueError("Missing required keys in response")
        if not isinstance(result["relevant_offerings"], list):
            result["relevant_offerings"] = []
        result["relevant_offerings"] = [
            offering for offering in result["relevant_offerings"]
            if offering in offerings_list
        ]
        if not isinstance(result["known_partners"], list):
            result["known_partners"] = []
        if not isinstance(result["similar_to_morek"], list):
            result["similar_to_morek"] = []
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
            "known_partners": [],
            "similar_to_morek": [],
            "relevant_offerings": [],
            "reasoning": str(e)
        }

@st.cache_data(show_spinner=False)
def label_company_cached(company_name):
    return label_company(company_name)

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload Delegate List PDF", type=["pdf"])

CACHE_PATH = "gow2025_labeled_companies.csv"
display_df = pd.DataFrame()  # Safe default

def highlight_partner(val):
    if val is True or val == True or str(val).lower() == "true":
        return 'background-color: #d4f7d4; font-weight: bold;'
    return ''

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
    else:
        st.dataframe(df)

    # Try to load cached results
    if os.path.exists(CACHE_PATH):
        cached_df = pd.read_csv(CACHE_PATH)
        # Merge on Company
        df = df.merge(cached_df, on="Company", how="left", suffixes=("", "_cached"))
    else:
        cached_df = pd.DataFrame()

    test_mode = st.toggle("🧪 Test Mode (Process only first 5 companies)", value=True)
    
    if st.button("🔍 Label Companies with GPT"):
        st.info("Calling GPT to label each unique company...")
        progress = st.progress(0)
        result_map = {}
        companies = df["Company"].unique()
        
        if test_mode:
            companies = companies[:5]
            st.warning("🧪 Test Mode: Processing only first 5 companies")

        # Only process companies not already labeled
        already_labeled = set(cached_df["Company"]) if not cached_df.empty else set()
        to_label = [c for c in companies if c not in already_labeled]

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

        # Use cached results for already labeled companies
        for i, company in enumerate(to_label):
            with st.spinner(f"Labeling {company}..."):
                result_map[company] = label_company_cached(company)
                progress.progress((i + 1) / max(1, len(to_label)))
                time.sleep(1.1)

        loading_placeholder.empty()  # Remove the custom spinner when done

        # Build new labeled DataFrame
        new_labeled = []
        for company in to_label:
            res = result_map.get(company, {})
            res["Company"] = company
            new_labeled.append(res)
        new_labeled_df = pd.DataFrame(new_labeled)

        # Combine with cached
        if not cached_df.empty:
            labeled_df = pd.concat([cached_df, new_labeled_df], ignore_index=True)
        else:
            labeled_df = new_labeled_df

        # Save to disk
        labeled_df.to_csv(CACHE_PATH, index=False)

        # Merge with main df for display
        df = df.merge(labeled_df, on="Company", how="left", suffixes=("", "_labeled"))

        # Attach results to DataFrame
        df["Industry"] = df["Company"].map(lambda x: result_map.get(x, {}).get("industry", ""))
        df["Company Type"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_type", ""))
        df["Business Model"] = df["Company"].map(lambda x: result_map.get(x, {}).get("business_model", ""))
        df["Company Size"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_size", ""))
        df["Company Age"] = df["Company"].map(lambda x: result_map.get(x, {}).get("company_age", ""))
        df["Potential Partner"] = df["Company"].map(lambda x: result_map.get(x, {}).get("potential_partner", False))
        df["Partner Reasoning"] = df["Company"].map(lambda x: result_map.get(x, {}).get("partner_reasoning", ""))
        df["Known Partners"] = df["Company"].map(lambda x: ", ".join(result_map.get(x, {}).get("known_partners", [])))
        df["Similar to Morek"] = df["Company"].map(
            lambda x: ", ".join([
                c for c in result_map.get(x, {}).get("similar_to_morek", [])
                if c.strip().lower() != "morek engineering"
            ])
        )
        df["Relevant Offerings"] = df["Company"].map(
            lambda x: ", ".join(result_map.get(x, {}).get("relevant_offerings", []))
        )
        df["Reasoning"] = df["Company"].map(lambda x: result_map.get(x, {}).get("reasoning", ""))

        # Add a filter to show only potential partners
        show_partners = st.checkbox("Show only potential partners", value=False)
        display_df = df[df["Potential Partner"]] if show_partners else df

        st.success("✅ Labeling complete.")
        st.dataframe(
            display_df.style.applymap(highlight_partner, subset=["Potential Partner"]),
            use_container_width=True
        )

    # --- Visualization Tabs ---
    tab1, tab2, tab3 = st.tabs(["Summary Graphs", "Company Cards", "Full Table"])

    # --- Summary Graphs ---
    with tab1:
        st.header("Summary Graphs")
        # Industry distribution
        if not display_df.empty:
            st.subheader("Companies by Industry")
            industry_counts = display_df["Industry"].value_counts().reset_index()
            industry_counts.columns = ["Industry", "Count"]
            st.altair_chart(
                alt.Chart(industry_counts).mark_bar().encode(
                    x=alt.X("Industry", sort="-y"),
                    y="Count",
                    tooltip=["Industry", "Count"]
                ).properties(height=300),
                use_container_width=True
            )

            st.subheader("Companies by Size")
            size_counts = display_df["Company Size"].value_counts().reset_index()
            size_counts.columns = ["Company Size", "Count"]
            st.altair_chart(
                alt.Chart(size_counts).mark_bar().encode(
                    x=alt.X("Company Size", sort="-y"),
                    y="Count",
                    tooltip=["Company Size", "Count"]
                ).properties(height=200),
                use_container_width=True
            )

            st.subheader("Potential Partners Distribution")
            partner_counts = display_df["Potential Partner"].value_counts().reset_index()
            partner_counts.columns = ["Potential Partner", "Count"]
            st.altair_chart(
                alt.Chart(partner_counts).mark_arc(innerRadius=40).encode(
                    theta="Count",
                    color="Potential Partner",
                    tooltip=["Potential Partner", "Count"]
                ).properties(height=200),
                use_container_width=True
            )

            st.subheader("Most Common Relevant Offerings")
            offerings_series = display_df["Relevant Offerings"].str.split(", ").explode()
            offerings_counts = offerings_series.value_counts().reset_index()
            offerings_counts.columns = ["Offering", "Count"]
            st.altair_chart(
                alt.Chart(offerings_counts).mark_bar().encode(
                    x=alt.X("Offering", sort="-y"),
                    y="Count",
                    tooltip=["Offering", "Count"]
                ).properties(height=200),
                use_container_width=True
            )
        else:
            st.info("No data to display.")

    # --- Company Cards ---
    with tab2:
        st.header("Company Cards")
        if not display_df.empty:
            for i in range(0, len(display_df), 3):
                cols = st.columns(3)
                for j, (_, row) in enumerate(display_df.iloc[i:i+3].iterrows()):
                    with cols[j]:
                        st.markdown(f"""
                            <div style='border:1px solid #ddd; border-radius:10px; padding:1em; margin-bottom:1em; background-color:{'#d4f7d4' if row['Potential Partner'] else '#fff'};'>
                                <h4>{row['Company']}</h4>
                                <b>Industry:</b> {row['Industry']}<br>
                                <b>Type:</b> {row['Company Type']}<br>
                                <b>Size:</b> {row['Company Size']}<br>
                                <b>Age:</b> {row['Company Age']}<br>
                                <b>Business Model:</b> {row['Business Model']}<br>
                                <b>Potential Partner:</b> {'✅' if row['Potential Partner'] else '❌'}<br>
                                <b>Relevant Offerings:</b> {row['Relevant Offerings']}<br>
                                <b>Known Partners:</b> {row['Known Partners']}<br>
                                <b>Similar to Morek:</b> {row['Similar to Morek']}<br>
                                <details><summary><b>Reasoning</b></summary>{row['Reasoning']}</details>
                                <details><summary><b>Partner Reasoning</b></summary>{row['Partner Reasoning']}</details>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No data to display.")

    # --- Full Table ---
    with tab3:
        st.header("Full Table")
        if "Potential Partner" in display_df.columns:
            st.dataframe(
                display_df.style.applymap(highlight_partner, subset=["Potential Partner"]),
                use_container_width=True
            )
        else:
            st.dataframe(display_df, use_container_width=True)

    # --- Download ---
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="gow2025_delegates_classified.csv")

else:
    st.info("Upload a GOW 2025 delegate list PDF to begin.")
