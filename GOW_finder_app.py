import streamlit as st
import pandas as pd
import pdfplumber
import openai
import json
import time
import altair as alt
import os
import pathlib
import concurrent.futures
from typing import Dict, List, Any

# --- Morek Branding ---
st.markdown(
    """
    <style>
    body {
        background-color: #fff !important;
        color: #002b45 !important;
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
    }
    .stApp {
        background-color: #fff !important;
    }
    .st-bb, .st-c3, .st-c6, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        color: #002b45 !important;
    }
    .stButton>button {
        background-color: #ffd600 !important;
        color: #002b45 !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.2em;
    }
    .stTextInput>div>div>input {
        background-color: #e6f2fa !important;
        color: #002b45 !important;
        border-radius: 6px;
    }
    .stDataFrame, .stTable {
        background-color: #e6f2fa !important;
        color: #002b45 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #00395c !important;
        color: #ffd600 !important;
        font-weight: bold;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffd600 !important;
        color: #002b45 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffd600 !important;
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
    }
    .morek-logo {
        width: 220px;
        margin-bottom: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Morek Logo Header ---
col1, col2 = st.columns([1, 6])
with col1:
    logo_path = "morek_logo.jpg"
    if pathlib.Path(logo_path).exists():
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("Logo file 'morek_logo.jpg' not found. Please add it to the project directory.")
with col2:
    st.markdown("<span style='font-size:2.2em; font-weight: bold; color: #ffd600;'>COMPANY SEARCH ENGINE</span>", unsafe_allow_html=True)

# --- Login ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    with st.form("login_form"):
        st.markdown("## Login Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "Morekian" and password == "Morek2025!":
                st.session_state['authenticated'] = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()

# --- Settings ---
# st.title("📄 COMPANY SEARCH ENGINE")

offerings_list = [
    "naval architecture support",
    "temporary mooring design",
    "detailed mooring design",
    "pre-FEED mooring and dynamic cable design",
    "seafastening",##
    "marine operations planning"
]

client = openai.OpenAI(api_key=st.secrets["openai_api_key"])#

def label_company(company_name):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
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
    "relevant_offerings": ["List of relevant services from: {', '.join(offerings_list)}"],
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
            response = client.chat.completions.create(
                model="gpt-4.1-mini",  # Using GPT-4.1-mini for better accuracy
                messages=[
                    {"role": "system", "content": "You are a precise and accurate marine engineering industry labeler. Always respond with valid JSON."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=60  # Increased timeout
            )
            
            # Extract and validate the response
            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON response for {company_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
            
            # Validate the response structure
            required_keys = [
                "industry", "company_type", "business_model", "company_size", "company_age",
                "potential_partner", "partner_reasoning", "known_partners", "similar_to_morek",
                "relevant_offerings", "reasoning"
            ]
            
            # Ensure all required keys exist with default values if missing
            for key in required_keys:
                if key not in result:
                    if key == "potential_partner":
                        result[key] = False
                    elif key in ["known_partners", "similar_to_morek", "relevant_offerings"]:
                        result[key] = []
                    else:
                        result[key] = "Unknown"
            
            # Ensure lists are actually lists
            for key in ["known_partners", "similar_to_morek", "relevant_offerings"]:
                if not isinstance(result[key], list):
                    result[key] = []
            
            # Filter relevant offerings
            result["relevant_offerings"] = [
                offering for offering in result["relevant_offerings"]
                if offering in offerings_list
            ]
            
            # Ensure boolean value for potential_partner
            result["potential_partner"] = bool(result["potential_partner"])
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            st.error(f"Error labeling {company_name} after {max_retries} attempts: {str(e)}")
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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def label_company_cached(company_name):
    return label_company(company_name)

def extract_companies(text: str) -> List[str]:
    """Extract company names from PDF text using GPT."""
    try:
        prompt = f"""
Extract a list of company names from the following text. Return only the company names, one per line.
Ignore any other information like names, job titles, or other text.

Text:
{text}

Return only the company names, one per line. Do not include any other text or explanation.
"""
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Using GPT-4.1-mini for better accuracy
            messages=[
                {"role": "system", "content": "You are a precise company name extractor. Return only company names, one per line."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Extract company names from response
        companies = [
            line.strip() for line in response.choices[0].message.content.split('\n')
            if line.strip() and not line.strip().startswith(('Here', 'The', 'I', 'This'))
        ]
        
        return companies
    except Exception as e:
        st.error(f"Error extracting companies: {str(e)}")
        return []

def clean_company_name(company: str) -> str:
    """Clean and standardize company names."""
    if not company:
        return ""
    
    # Convert to string and strip whitespace
    company = str(company).strip()
    
    # Remove common suffixes and standardize
    suffixes = [
        " ltd", " limited", " llc", " inc", " corporation", " corp", " plc", " gmbh",
        " pty", " pvt", " private", " company", " co", " group", " holdings", " holding"
    ]
    
    # Convert to lowercase for processing
    company_lower = company.lower()
    
    # Remove suffixes
    for suffix in suffixes:
        if company_lower.endswith(suffix):
            company = company[:-len(suffix)]
    
    # Remove extra whitespace and standardize spacing
    company = " ".join(company.split())
    
    # Capitalize each word
    company = company.title()
    
    return company

def highlight_partner(val):
    if val is True or val == True or str(val).lower() == "true":
        return 'background-color: #d4f7d4; font-weight: bold;'
    return ''

def safe_display_dataframe(df, use_styling=True):
    """Safely display a DataFrame with optional styling."""
    if df.empty:
        st.info("No data to display.")
        return
    
    # Ensure required columns exist
    if use_styling and "Potential Partner" not in df.columns:
        df["Potential Partner"] = False
    
    try:
        if use_styling:
            styled_df = df.style.applymap(
                highlight_partner,
                subset=["Potential Partner"]
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Error displaying table: {str(e)}")
        st.dataframe(df, use_container_width=True)  # Fallback to unstyled display

# Initialize session state for data storage
if 'labeled_data' not in st.session_state:
    st.session_state.labeled_data = pd.DataFrame()

# --- File Upload ---
st.markdown("## 📄 Upload Delegate List")
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    # Extract text from PDF
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Extract companies using GPT
    companies = extract_companies(text)
    
    if companies:
        # Clean and deduplicate company names
        cleaned_companies = [clean_company_name(company) for company in companies]
        unique_companies = list(dict.fromkeys(cleaned_companies))  # Preserve order while removing duplicates
        
        st.success(f"✅ Found {len(unique_companies)} unique companies!")
        
        # Create base dataframe
        base_df = pd.DataFrame({
            "Company": unique_companies,
            "Industry": "",
            "Company Type": "",
            "Business Model": "",
            "Company Size": "",
            "Company Age": "",
            "Potential Partner": False,
            "Partner Reasoning": "",
            "Known Partners": "",
            "Similar to Morek": "",
            "Relevant Offerings": "",
            "Reasoning": ""
        })
        
        # Initialize display_df if not exists
        if 'display_df' not in st.session_state:
            st.session_state.display_df = base_df.copy()
        
        # Update display_df with any existing labeled data
        if not st.session_state.labeled_data.empty:
            for company in st.session_state.display_df["Company"].unique():
                if company in st.session_state.labeled_data["Company"].values:
                    cached_row = st.session_state.labeled_data[st.session_state.labeled_data["Company"] == company].iloc[0]
                    mask = st.session_state.display_df["Company"] == company
                    for col in cached_row.index:
                        if col != "Company":
                            st.session_state.display_df.loc[mask, col] = cached_row[col]

        # --- Label Companies ---
        if st.button("Label Companies with GPT"):
            # Get companies that haven't been labeled yet
            to_label = [c for c in unique_companies if c not in st.session_state.labeled_data["Company"].values]
            
            if not to_label:
                st.info("All companies have already been labeled. Use the search to view results.")
                st.stop()

            # Process companies in batches
            batch_size = 5 if test_mode else 10
            total = len(to_label)
            
            with st.spinner("Processing companies..."):
                for i in range(0, total, batch_size):
                    batch = to_label[i:i + batch_size]
                    new_labeled = []
                    
                    for company in batch:
                        try:
                            result = label_company(company)
                            result["Company"] = company
                            new_labeled.append(result)
                        except Exception as e:
                            st.error(f"Error processing {company}: {str(e)}")
                    
                    if new_labeled:
                        # Update session state with new results
                        new_df = pd.DataFrame(new_labeled)
                        st.session_state.labeled_data = pd.concat([st.session_state.labeled_data, new_df], ignore_index=True)
                        st.session_state.labeled_data = st.session_state.labeled_data.drop_duplicates(subset=["Company"], keep="last")
                        
                        # Update display_df
                        for result in new_labeled:
                            mask = st.session_state.display_df["Company"] == result["Company"]
                            for col in result.keys():
                                if col != "Company":
                                    st.session_state.display_df.loc[mask, col] = result[col]
                        
                        st.success(f"✅ Processed {len(new_labeled)} of {total} companies")
                    
                    # Small delay between batches
                    time.sleep(2)
            
            st.success("🎉 All companies processed!")
            st.rerun()

    # --- Search ---
    search_term = st.text_input("🔍 Search by name, job title, company:")
    
    # Add toggle for potential partners
    show_only_potential = st.toggle("👥 Show Only Potential Partners", value=False)
    
    if search_term or show_only_potential:
        # Create a mask for each column and combine them
        mask = pd.Series(False, index=st.session_state.display_df.index)
        
        # Apply search filter if search term exists
        if search_term:
            for column in st.session_state.display_df.columns:
                mask |= st.session_state.display_df[column].astype(str).str.contains(search_term, case=False, na=False)
        
        # Apply potential partners filter
        if show_only_potential:
            potential_mask = st.session_state.display_df["Potential Partner"].astype(str).str.lower() == "true"
            mask = mask & potential_mask if search_term else potential_mask
        
        filtered = st.session_state.display_df[mask]
        st.write(f"🔎 Found {len(filtered)} result(s):")
        safe_display_dataframe(filtered, use_styling=True)
    else:
        safe_display_dataframe(st.session_state.display_df, use_styling=True)

    test_mode = st.toggle("🧪 Test Mode (Process only first 5 companies)", value=True)

    # --- Visualization Tabs ---
    tab1, tab2, tab3 = st.tabs(["Summary Graphs", "Company Cards", "Full Table"])

    # --- Summary Graphs ---
    with tab1:
        st.header("Summary Graphs")
        # Load complete dataset for visualization
        complete_df = st.session_state.display_df

        if not complete_df.empty:
            # Industry distribution
            st.subheader("Companies by Industry")
            industry_counts = complete_df["Industry"].fillna("Unknown").value_counts().reset_index()
            industry_counts.columns = ["Industry", "Count"]
            st.altair_chart(
                alt.Chart(industry_counts).mark_bar().encode(
                    x=alt.X("Industry", sort="-y"),
                    y="Count",
                    tooltip=["Industry", "Count"]
                ).properties(height=300),
                use_container_width=True
            )

            # Company size distribution
            st.subheader("Companies by Size")
            size_counts = complete_df["Company Size"].fillna("Unknown").value_counts().reset_index()
            size_counts.columns = ["Company Size", "Count"]
            st.altair_chart(
                alt.Chart(size_counts).mark_bar().encode(
                    x=alt.X("Company Size", sort="-y"),
                    y="Count",
                    tooltip=["Company Size", "Count"]
                ).properties(height=200),
                use_container_width=True
            )

            # Potential partners distribution
            st.subheader("Potential Partners Distribution")
            partner_counts = complete_df["Potential Partner"].fillna(False).value_counts().reset_index()
            partner_counts.columns = ["Potential Partner", "Count"]
            st.altair_chart(
                alt.Chart(partner_counts).mark_arc(innerRadius=40).encode(
                    theta="Count",
                    color="Potential Partner",
                    tooltip=["Potential Partner", "Count"]
                ).properties(height=200),
                use_container_width=True
            )

            # Relevant offerings distribution
            st.subheader("Most Common Relevant Offerings")
            offerings_series = complete_df["Relevant Offerings"].fillna("").str.split(", ").explode()
            offerings_counts = offerings_series[offerings_series != ""].value_counts().reset_index()
            offerings_counts.columns = ["Offering", "Count"]
            if not offerings_counts.empty:
                st.altair_chart(
                    alt.Chart(offerings_counts).mark_bar().encode(
                        x=alt.X("Offering", sort="-y"),
                        y="Count",
                        tooltip=["Offering", "Count"]
                    ).properties(height=200),
                    use_container_width=True
                )
            else:
                st.info("No offerings data available.")
        else:
            st.info("No data to display.")

    # --- Company Cards ---
    with tab2:
        st.header("Company Cards")
        if not complete_df.empty:
            for i in range(0, len(complete_df), 3):
                cols = st.columns(3)
                for j, (_, row) in enumerate(complete_df.iloc[i:i+3].iterrows()):
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div style='border:1px solid #ddd; border-radius:10px; padding:1em; margin-bottom:1em; background-color:{'#d4f7d4' if row.get('Potential Partner', False) else '#fff'};'>
                                <h4>{row['Company']}</h4>
                                <b>Industry:</b> {row.get('Industry', 'Unknown')}<br>
                                <b>Type:</b> {row.get('Company Type', 'Unknown')}<br>
                                <b>Size:</b> {row.get('Company Size', 'Unknown')}<br>
                                <b>Age:</b> {row.get('Company Age', 'Unknown')}<br>
                                <b>Business Model:</b> {row.get('Business Model', 'Unknown')}<br>
                                <b>Potential Partner:</b> {'✅' if row.get('Potential Partner', False) else '❌'}<br>
                                <b>Relevant Offerings:</b> {row.get('Relevant Offerings', '')}<br>
                                <b>Known Partners:</b> {row.get('Known Partners', '')}<br>
                                <b>Similar to Morek:</b> {row.get('Similar to Morek', '')}<br>
                                <details><summary><b>Reasoning</b></summary>{row.get('Reasoning', '')}</details>
                                <details><summary><b>Partner Reasoning</b></summary>{row.get('Partner Reasoning', '')}</details>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.info("No data to display.")

    # --- Full Table ---
    with tab3:
        st.header("Full Table")
        safe_display_dataframe(complete_df, use_styling=True)

    # --- Download ---
    st.download_button("⬇️ Download CSV", complete_df.to_csv(index=False), file_name="gow2025_delegates_classified.csv")

else:
    st.info("Upload a GOW 2025 delegate list PDF to begin.")
