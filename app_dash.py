from flask import request
# app_dash.py
# fyi you're gonna see a TON of poorly formatted code here, but it WORKS baby

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ALL, ctx # Import ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import numpy as np
import json # Needed for GeoJSON potentially, though px might handle states directly

# --- NEW IMPORTS for Regression ---
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
# --- End NEW IMPORTS ---

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "assets/bootstrap.min.css", # --- adding path to my local copy to prevent CDN caching issues
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
], assets_folder='assets') # Removed suppress_callback_exceptions=True
server = app.server

@server.after_request
def add_header(response):
    # Get the base path for assets using Dash's recommended method
    assets_base_path = app.get_asset_url('') # This will be something like '/assets/'

    # Check if the request path starts with what Dash uses for assets
    is_asset_request = request.path.startswith(assets_base_path)

    if response.mimetype == 'text/html' or \
       (response.mimetype == 'text/css' and is_asset_request):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# --- Constants and Styles ---
DATA_FILE = "kizik_generated_sim_data_FINAL.csv"
IMAGE_FILE_PATH_RELATIVE = "assets/final flowchart.png"
# Optional: Add path to US States GeoJSON if needed for regional maps,
# but Plotly Express often handles state abbreviations directly.
# GEOJSON_FILE = "assets/us-states.json"

XM_COLORS = {
    'x1': '#04C9CE', 'x2': '#00B4EE', 'x3': '#038FEB', 'x4': '#0768DD',
    'x5': '#5F1AE5', 'x6': '#A54AF4', 'text': '#2B2B2B',
    # Colors for segment charts - ensure all segments are covered
    'Premium Enthusiasts': '#5F1AE5',
    'Value Seekers': '#00B4EE',
    'Interested Pragmatists': '#0768DD',
    'Curious Skeptics': '#21DBAA', # Using a distinct color
    'Traditionalists': '#E672FF' # Using another distinct color
}

# Define VW Colors based on the LATEST request (using INVERTED labels)
# Assigning Purple, Blue, Green, Red to the standard labels based on the EXAMPLE IMAGE
VW_CHART_COLORS_INVERTED = {
    'Too Inexpensive': 'green', # Descending
    'Inexpensive': 'blue',     # Descending (equivalent to "Good Value" in standard)
    'Expensive': 'purple',     # Ascending
    'Too Expensive': 'red'       # Ascending
}


# Updated segment info component with final content
SEGMENT_INFO = {
    "Curious Skeptics": {
        "number": 4,
        "title": "Curious Skeptics",
        "priorities": [
            "Lower value for hands-free (3.2/5), lower priority for convenience (3/5).",
            "Moderate importance: Comfort, Style, Durability, Price, Brand (all 4/5)."
        ],
        "basic_description": "Curious Skeptics are open to new products but are cautious. Convenience isn't their top driver; they seek balanced comfort, style, durability, and price, with moderate brand influence. They need convincing of a product's value beyond hype.",
        "segment_example_profile": "Priya, a practical high school science teacher, needs comfortable shoes for all-day classroom wear. While aware of hands-free tech, she questions its real utility for her, prioritizing sensible style and durability. She'd consider Kiziks if clearly beneficial and reasonably priced, after extensive research and review comparisons.",
        "data_signature": {
            "WTP/VW Charts": "Exhibit lower-mid-range price thresholds.",
            "GG Chart": "Shows smaller WTP uplift post-explanation.",
            "Top 3 Drivers": "Style, Brand, Comfort, Price are prominent; HandsFree less so.",
            "Regression Plot": "Weaker WTP relationships; lower average ValuePerceptionPost.",
            "Primary Usage": "Usage patterns are less distinct than other segments."
        }
    },
    "Value Seekers": {
        "number": 2,
        "title": "Value Seekers",
        "priorities": [
            "Highest priority for Price (5/5), high priority for Durability, Comfort (both 5/5).",
            "Lower importance: Style (3/5), HandsFree Convenience (3/5), Brand (2/5).",
            "Moderate simulated value for hands-free (3.8/5)."
        ],
        "basic_description": "Value Seekers prioritize functionality and longevity at the best possible price. While they appreciate comfort, they are less swayed by trends or brand prestige. Hands-free convenience is a secondary concern unless it offers clear, practical benefits without a significant price increase.",
        "segment_example_profile": "Mark, a retired handyman, seeks durable, comfortable, and affordable shoes for everyday tasks and light outdoor work. He's skeptical of \"fancy\" features but might be swayed by hands-free if the price is right and it genuinely makes life easier.",
        "data_signature": {
            "WTP/VW Charts": "Exhibit the lowest price thresholds.",
            "GG Chart": "Shows a smaller percentage increase in WTP post-explanation.",
            "Top 3 Drivers": "Price, Durability, and Comfort are most frequent.",
            "Regression Plot": "Associated with negative coefficients for lower IncomeBrackets.",
            "Primary Usage": "Concentrated in Everyday casual wear and Work/Office."
        }
    },
    "Interested Pragmatists": {
        "number": 3,
        "title": "Interested Pragmatists",
        "priorities": [
            "Balanced high priorities: Comfort (5/5), HandsFree Convenience (4/5).",
            "Style, Durability, Price (all 4/5). Hands-free appeal and high simulated likelihood for repeat purchases (4.0/5).",
        ],
        "basic_description": "Interested Pragmatists are practical consumers who see the appeal of innovative features like hands-free technology, especially if it enhances comfort and daily convenience. They balance this interest with considerations of style, durability, and fair pricing. They are likely to adopt new tech if its benefits are clear and well-integrated.",
        "segment_example_profile": "Alejandra, a busy working mom and graphic designer, values shoes that simplify her hectic routine without sacrificing style or comfort. She's intrigued by hands-free tech for its convenience, especially when juggling kids and work, and is willing to pay a reasonable premium for it.",
        "data_signature": {
            "WTP/VW Charts": "Exhibit mid-range price thresholds.",
            "GG Chart": "Shows a significant positive WTP uplift post-explanation.",
            "Top 3 Drivers": "Displays a balanced mix of frequently mentioned drivers.",
            "Regression Plot": "Associated with positive ValuePerceptionPost; may be baseline.",
            "Primary Usage": "Shows a diverse usage profile across multiple scenarios."
        }
    },
    "Premium Enthusiasts": {
        "number": 1,
        "title": "Premium Enthusiasts",
        "priorities": [
            "Highest value for hands-free (4.5/5), highest priority for convenience (5/5).",
            "Emphasize: Style (5/5), Comfort (5/5). Lowest importance for Price (3/5).",
            "Highest simulated likelihood for repeat purchases (4.5/5)."
        ],
        "basic_description": "Premium Enthusiasts are early adopters who value innovation, style, and superior comfort. They are willing to pay more for products that offer unique benefits and a premium experience, such as Kizik's hands-free technology. Price is a secondary concern to quality and cutting-edge features.",
        "segment_example_profile": "Naomi, a tech executive who travels frequently, seeks stylish, comfortable, and innovative footwear. She highly values the convenience of hands-free shoes for airport security and a fast-paced lifestyle, and is willing to invest in premium brands that offer such features.",
        "data_signature": {
            "WTP/VW Charts": "Exhibit the highest price thresholds.",
            "GG Chart": "Shows the largest percentage increase in WTP post-explanation.",
            "Top 3 Drivers": "HandsFree Convenience, Style, Brand/Comfort prominent; Price less so.",
            "Regression Plot": "Correlates with high ValuePerceptionPost and higher IncomeBrackets.",
            "Primary Usage": "Higher proportions for Style/Fashion statement and Travel."
        }
    },
    "Traditionalists": {
        "number": 5,
        "title": "Traditionalists",
        "priorities": [
            "Lowest value for hands-free (2.5/5), very low priority for convenience (1/5).",
            "High priority: Comfort (5/5), Durability (5/5), Price (5/5).",
            "Lowest simulated likelihood for repeat purchases (2.0/5).",
        ],
        "basic_description": "Traditionalists prefer proven, familiar footwear attributes like comfort, durability, and value for money. They are often skeptical of new technologies like hands-free features, viewing them as unnecessary or faddish. Their purchasing decisions are driven by practicality and established habits.",
        "segment_example_profile": "Martin, a retired librarian, prefers classic, comfortable shoes from brands he trusts. He values durability and a fair price above all else and sees little need for hands-free technology, finding traditional slip-ons or lace-ups perfectly adequate.",
        "data_signature": {
            "WTP/VW Charts": "Exhibit the lowest or second-lowest price thresholds.",
            "GG Chart": "Shows minimal or no WTP uplift post-explanation.",
            "Top 3 Drivers": "Comfort, Durability, Price most frequent; HandsFree rarely mentioned.",
            "Regression Plot": "Associated with negative coefficients for older AgeGroup; lowest ValuePerceptionPost.",
            "Primary Usage": "Higher for Specific activity along with Everyday casual."
        }
    }
}
# Define the order for the tabs to match the screenshot
SEGMENT_ORDER = ["Curious Skeptics", "Value Seekers", "Interested Pragmatists", "Premium Enthusiasts", "Traditionalists"]

# --- NEW: Define Limitations Content ---
limitations_content = [
    {
        "id": "limitation-sim-data",
        "title": "Parameter-Informed Simulation, Not True Synthetic/Real Data",
        "content": """**Critique:** This is by far the biggest limitation. The relationships between the variables are defined by the parameters I set with Claude and Gemini. Although they have real-world roots (Census, store locations, correctly modeled WTP formulas etc), it's still a simulation based on assumptions and those parameters, not a true, in-field research project informed by responses from actual people. It can demonstrate good methodology and design, but it will ultimately fail to capture truly unique or unexpected patterns emerging from the complexity of responses generated by… real, complex people like you and me.\n\nKiziks to some real-life segments (or our grandparents!) might as well be just the same as the Skechers they've worn for years, or the Skechers that just recently copied Kizik's hands-free tech. Nuances like that can't be fully captured in a simulation like this.\n\n**Versus:** Additionally, true synthetic data - as I've mentioned before - is derived from training GANs on actual (generally private) survey data to learn complex new relationships that can't be fully gleaned from a parameterized simulation like this."""
    },
    {
        "id": "limitation-feature-exp",
        "title": "\"Conceptual\" Feature Explanation",
        "content": """**Critique:** My simulation models the effect of the explanation through an assumed uplift factor but obviously doesn't capture the actual "aha moment" of seeing or experiencing a demo. Different demos can have different impacts across different segments, but because of these inherent limitations in our model, the simulation applies a relatively uniform boost to the uplift factor.\n\n**Versus:** In-field testing could include showing a gif or video, or if in-store, an actual hands-free "hands on" demo and measuring the response, capturing authentic reactions."""
    },
    {
        "id": "limitation-awareness-proxy",
        "title": "Simplified Awareness Proxy (BrandAwarenessProxy_DistWt)",
        "content": """**Critique:** This one is fairly simple. Our current model takes into account store locations, flagship vs retail, and includes distance decay and an adjacency bonus, but it cearly falls short of the complex nuances of a real-world survey. Brand awareness isn't just physical proximity: it's exposure to ads - physical and digital, word of mouth, paid social, time in market, and more. Without internal Kizik brand data on factors like those, it's a crude proxy that might correlate with some aspects of those factors.\n\n**Versus:** Real-world research could use direct survey questions ("How familiar are you with Kizik?") or measure social media metrics, etc."""
    },
    {
        "id": "limitation-competitive",
        "title": "Lack of a Competitive Landscape",
        "content": """**Critique:** We applied a 20% competitive adjustment factor, reducing WTP by 20% (supported by research from Nielsen, etc.). Obviously, real-world competitive dynamics are modeled with much more than slapping a simple adjustment factor onto the data. WTP depends extensively on what alternatives are actually available to the respondent at that moment, their prices, as well as their features.\n\n**Versus:** More specific survey questions or methodologies could help capture these tradeoffs more accurately - maybe a Conjoint Analysis (had to reject it here)."""
    },
    {
        "id": "limitation-segmentation",
        "title": "Simplified Segmentation Logic",
        "content": """**Critique:** New segments are not discovered in our simulation. The segments used are assigned probabilistically weighted profiles from the simulation code, but don't represent any emergence of some novel, undiscovered segment. Respondents are assigned to pre-baked profiles\n\n**Versus:** In-field research will use different analytical tools to identify naturally occurring segments. Cluster analysis, anyone?"""
    }
]
# --- END: Define Limitations Content ---

# --- Helper function to generate segment content ---
def create_segment_content(segment_name):
    """Generates the HTML structure for the segment content display area."""
    if segment_name not in SEGMENT_INFO:
        return html.P("Segment details not found.")

    info = SEGMENT_INFO[segment_name]
    priorities_list = info.get("priorities", [])
    basic_desc = info.get("basic_description", "N/A")
    example_profile_desc = info.get("segment_example_profile", "N/A")
    data_sig_dict = info.get("data_signature", {})

    content_children = []

    # Priorities
    content_children.append(html.P(html.Strong("Priorities:"), style={'margin-bottom': '0.5rem', 'margin-top': '0'}))
    if priorities_list:
        content_children.append(
            html.Ul([html.Li(item) for item in priorities_list], style={'padding-left': '20px', 'margin-bottom': '1rem', 'font-size': '15px'})
        )

    # Description
    content_children.append(html.P(html.Strong("Description:"), style={'margin-bottom': '0.5rem'}))
    content_children.append(dcc.Markdown(basic_desc, className="text-content", style={'margin-top': '0', 'font-size': '15px'}))

    # Segment Example Profile
    content_children.append(html.P(html.Strong("Segment Example Profile:"), style={'margin-bottom': '0.5rem'}))
    content_children.append(dcc.Markdown(example_profile_desc, className="text-content", style={'margin-top': '0', 'font-size': '15px'}))

    # Data Signature
    content_children.append(html.P(html.Strong("Data Signature:"), style={'margin-bottom': '0.5rem'}))
    if data_sig_dict:
        data_sig_items = []
        # Use the specific key order
        data_sig_key_order = ["WTP/VW Charts", "GG Chart", "Top 3 Drivers", "Regression Plot", "Primary Usage"]
        for key in data_sig_key_order:
            if key in data_sig_dict: # Check if key exists to prevent errors
                data_sig_items.append(html.Li([html.Strong(f"{key}: "), data_sig_dict[key]]))
        content_children.append(html.Ul(data_sig_items, style={'padding-left': '20px', 'margin-bottom': '1rem', 'font-size': '15px'}))    
    return html.Div(content_children)


# --- HELPER FUNCTION for Limitation Content ---
def create_limitation_content(limitation_id, content_list):
    """Generates the HTML structure for a specific limitation content panel."""
    for item in content_list:
        if item['id'] == limitation_id:
            # Use dcc.Markdown to render content, apply consistent styling
            return dcc.Markdown(
                item['content'],
                className="text-content", # Reuse existing class for consistency
                style={'margin-top': '0', 'margin-bottom': '0', 'font-size': '15px'} # Reset margins for display area
            )
    return html.P("Limitation details not found.")

STANDARD_GRAPH_CONFIG = {'displayModeBar': True, 'displaylogo': False}
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "1.5rem 1rem",
    "background-color": "white",
    "overflow-y": "auto",
    "z-index": 1000,
    "box-shadow": "0 2px 10px rgba(0,0,0,0.05)",
    "outline": "none",
    "border-right": "none",
    "direction": "ltr"
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "0", # Changed from "1rem" to "0"
    "padding": "2rem 1rem",
    "overflow-y": "auto",
    "height": "100vh",
    "display": "flex",
    "flex-direction": "column",
    "align-items": "center"
}

# --- Data Loading Function ---
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            # Ensure relevant columns are numeric, coercing errors
            prob_cols = [col for col in df.columns if 'Prob_' in col]
            intent_cols = [col for col in df.columns if 'Intent_' in col]
            importance_cols = [col for col in df.columns if 'Importance_' in col]
            vw_cols = ['VW_TooCheap', 'VW_Bargain', 'VW_Expensive', 'VW_TooExpensive', 'ValuePerceptionPost']
            dist_cols = ['DistNearestRetailerMiles', 'DistNearestFlagshipMiles','BrandAwarenessProxy_DistWt']
            # Add group identifiers to numeric check
            group_cols = ['IsControlGroup', 'SawFeatureExplanation']

            all_numeric_cols = prob_cols + intent_cols + importance_cols + vw_cols + dist_cols + group_cols
            for col in all_numeric_cols:
                 if col in df.columns:
                     # Convert safely, leave existing NaNs as NaNs
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                 else:
                     print(f"Warning: Column {col} not found during numeric conversion.")

            print(f"Data loaded successfully from {file_path} with {len(df)} rows.")
            # Clean segment names if necessary (e.g., remove leading/trailing spaces)
            if 'Segment' in df.columns:
                df['Segment'] = df['Segment'].str.strip()
            # Clean column names for statsmodels formula compatibility
            df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
            return df
        except Exception as e:
            print(f"Error loading CSV '{file_path}': {e}")
            return None
    else:
        print(f"Error: Data file not found at path: {file_path}")
        return None

# --- LOAD DATA GLOBALLY ---
print("Attempting to load data globally...")
app_data = load_data(DATA_FILE) # Call the function

if app_data is None:
    print("FATAL: Global data loading failed. App functionality will be limited.")
    # Consider how the app should behave if data is essential and fails to load.
    # For now, subsequent checks in callbacks will handle this.
else:
    print(f"Global data loaded successfully with {len(app_data)} rows.")
# --- END GLOBAL LOAD ---


# --- Helper Function for WTP Calculation ---
def calculate_wtp_series(df, price_points_dict, likelihood_threshold=50):
    """Calculates WTP series from probability columns."""
    wtp_results = []
    total_respondents = len(df)
    if total_respondents == 0:
        return pd.DataFrame({'Price': [], 'Willing (%)': []})

    for price, col_name in sorted(price_points_dict.items()):
         # Adjust col_name for cleaned column names if necessary (e.g., if they had special chars)
        cleaned_col_name = col_name.replace('[^A-Za-z0-9_]+', '')
        if cleaned_col_name in df.columns:
            df[cleaned_col_name] = pd.to_numeric(df[cleaned_col_name], errors='coerce').fillna(0)
            willing_count = df[df[cleaned_col_name] >= likelihood_threshold].shape[0]
            willing_percent = (willing_count / total_respondents) * 100 if total_respondents > 0 else 0
            wtp_results.append({'Price': price, 'Willing (%)': willing_percent})
        else:
            print(f"Warning: WTP Column {cleaned_col_name} (cleaned from {col_name}) not found.")
            wtp_results.append({'Price': price, 'Willing (%)': 0})
    return pd.DataFrame(wtp_results)

# --- Plotting Functions ---

# 1. Willingness To Pay (Gabor-Granger - Pre/Post, Segmented)
def create_wtp_gg_chart(segment_filter='All Segments'): # Removed df argument
    if app_data is None: # Check global
         return go.Figure().update_layout(title_text="Data unavailable for WTP chart", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty: # Check if the globally loaded data (now local df) is empty
        return go.Figure().update_layout(title_text="No data available for WTP chart", plot_bgcolor='white', paper_bgcolor='white')

    # Clean price point dict keys to match cleaned column names
    price_points_initial = {79: 'Initial_Prob_79', 99: 'Initial_Prob_99', 119: 'Initial_Prob_119', 139: 'Initial_Prob_139', 159: 'Initial_Prob_159', 179: 'Initial_Prob_179', 199: 'Initial_Prob_199'}
    price_points_post = {79: 'Post_Prob_79', 99: 'Post_Prob_99', 119: 'Post_Prob_119', 139: 'Post_Prob_139', 159: 'Post_Prob_159', 179: 'Post_Prob_179', 199: 'Post_Prob_199'}
    price_points_initial_clean = {p: k.replace('[^A-Za-z0-9_]+', '') for p, k in price_points_initial.items()}
    price_points_post_clean = {p: k.replace('[^A-Za-z0-9_]+', '') for p, k in price_points_post.items()}


    if segment_filter != 'All Segments':
        if 'Segment' not in df.columns:
             return go.Figure().update_layout(title_text="Error: 'Segment' column missing", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df[df['Segment'] == segment_filter].copy()
        if df_filtered.empty:
            return go.Figure().update_layout(title_text=f"No data for segment: {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')
        chart_title = f"Willingness to Pay: Pre vs. Post Feature Explanation ({segment_filter})"
    else:
        df_filtered = df.copy()
        chart_title = "Willingness to Pay: Pre vs. Post Feature Explanation (All Segments)"

    wtp_initial_df = calculate_wtp_series(df_filtered, price_points_initial_clean)
    wtp_post_df = calculate_wtp_series(df_filtered, price_points_post_clean)

    if wtp_initial_df.empty or wtp_post_df.empty:
        return go.Figure().update_layout(title_text=f"Not enough data to calculate WTP for {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wtp_initial_df['Price'], y=wtp_initial_df['Willing (%)'], mode='lines+markers', name='Pre-Explanation', line=dict(color=XM_COLORS['x2'], width=3, dash='dash'), marker=dict(symbol='circle', size=8)))
    fig.add_trace(go.Scatter(x=wtp_post_df['Price'], y=wtp_post_df['Willing (%)'], mode='lines+markers', name='Post-Explanation', line=dict(color=XM_COLORS['x5'], width=3), marker=dict(symbol='circle', size=8)))

    fig.update_layout(
        title={'text': chart_title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
        xaxis_title="Price Point ($)", yaxis_title="Respondents Willing to Pay (%)",
        yaxis_ticksuffix="%", yaxis_range=[0, 105], xaxis_tickprefix="$",
        plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif",
        font_color=XM_COLORS['text'], hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 2. Regional WTP Map
def create_regional_map(segment_filter='All Segments', wtp_price_point=139): # Removed df argument
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Regional Map", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty: # Check if the globally loaded data (now local df) is empty
        return go.Figure().update_layout(title_text="No data available for Regional Map", plot_bgcolor='white', paper_bgcolor='white')

    if segment_filter != 'All Segments':
        if 'Segment' not in df.columns:
             return go.Figure().update_layout(title_text="Error: 'Segment' column missing", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df[df['Segment'] == segment_filter].copy()
        if df_filtered.empty:
            return go.Figure().update_layout(title_text=f"No data for segment: {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')
        map_title = f"Regional Willingness to Pay at ${wtp_price_point} ({segment_filter})"
    else:
        df_filtered = df.copy()
        map_title = f"Regional Willingness to Pay at ${wtp_price_point} (All Segments)"

    # Adjust price col name for cleaned columns
    price_col_base = f'Post_Prob_{wtp_price_point}'
    price_col = price_col_base.replace('[^A-Za-z0-9_]+', '')
    if price_col not in df_filtered.columns:
         print(f"Error: Column {price_col} (cleaned from {price_col_base}) not found for map.")
         return go.Figure().update_layout(title_text=f"Error: Data missing for ${wtp_price_point}", plot_bgcolor='white', paper_bgcolor='white')

    likelihood_threshold = 50
    df_filtered[price_col] = pd.to_numeric(df_filtered[price_col], errors='coerce').fillna(0)

    state_wtp = df_filtered.groupby('State').agg(
        Total=('RespondentID', 'count'),
        WillingCount=(price_col, lambda x: (x >= likelihood_threshold).sum())
    ).reset_index()

    state_wtp = state_wtp[state_wtp['Total'] > 0]
    if state_wtp.empty:
        return go.Figure().update_layout(title_text=f"No regional WTP data for ${wtp_price_point} in {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')
    state_wtp['Willing (%)'] = (state_wtp['WillingCount'] / state_wtp['Total'] * 100).round(1)

    fig = px.choropleth(
        state_wtp, locations='State', locationmode='USA-states', color='Willing (%)',
        scope='usa', color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[0, max(1, state_wtp['Willing (%)'].max())], # Ensure range_color max is at least 1
        labels={'Willing (%)': '% Willing to Pay'}, hover_name='State',
        hover_data={'State': False, 'Willing (%)': ':.1f%', 'Total': True}
    )
    fig.update_layout(
        title={'text': map_title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
        geo=dict(bgcolor='rgba(0,0,0,0)'), paper_bgcolor='white', font_family="Inter, sans-serif",
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    return fig

# 3. Expansion Priority Matrix
def create_expansion_matrix(segment_filter='All Segments', wtp_price_point=139): # Removed df argument
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Expansion Matrix", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty: # Check if the globally loaded data (now local df) is empty
        return go.Figure().update_layout(title_text="No data available for Expansion Matrix", plot_bgcolor='white', paper_bgcolor='white')

    if segment_filter != 'All Segments':
        if 'Segment' not in df.columns:
             return go.Figure().update_layout(title_text="Error: 'Segment' column missing", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df[df['Segment'] == segment_filter].copy()
        if df_filtered.empty:
            return go.Figure().update_layout(title_text=f"No data for segment: {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')
        matrix_title = f"Expansion Priority Matrix ({segment_filter})"
    else:
        df_filtered = df.copy()
        matrix_title = "Expansion Priority Matrix (All Segments)"

    price_col_base = f'Post_Prob_{wtp_price_point}'
    price_col = price_col_base.replace('[^A-Za-z0-9_]+', '')
    awareness_col = 'BrandAwarenessProxy_DistWt' # Already cleaned

    if price_col not in df_filtered.columns or awareness_col not in df_filtered.columns:
         print(f"Error: Required columns missing for Expansion Matrix ({price_col}, {awareness_col}).")
         return go.Figure().update_layout(title_text="Error: Missing data for Matrix", plot_bgcolor='white', paper_bgcolor='white')

    likelihood_threshold = 50
    df_filtered[price_col] = pd.to_numeric(df_filtered[price_col], errors='coerce').fillna(0)
    df_filtered[awareness_col] = pd.to_numeric(df_filtered[awareness_col], errors='coerce').fillna(0)

    region_data = df_filtered.groupby('Region').agg(
        AvgAwareness=(awareness_col, 'mean'),
        Total=('RespondentID', 'count'),
        WillingCount=(price_col, lambda x: (x >= likelihood_threshold).sum())
    ).reset_index()

    region_data = region_data[region_data['Total'] > 5]
    if region_data.empty:
         return go.Figure().update_layout(title_text=f"Insufficient regional data for {segment_filter} (min 5 per region)", plot_bgcolor='white', paper_bgcolor='white')
    region_data['Willing (%)'] = (region_data['WillingCount'] / region_data['Total'] * 100).round(1)

    min_awareness = region_data['AvgAwareness'].min()
    max_awareness = region_data['AvgAwareness'].max()
    if max_awareness > min_awareness:
        region_data['AwarenessOpportunity'] = ((max_awareness - region_data['AvgAwareness']) / (max_awareness - min_awareness)) * 100
    else:
        region_data['AwarenessOpportunity'] = 50

    median_wtp = region_data['Willing (%)'].median()
    median_opportunity = region_data['AwarenessOpportunity'].median()

    fig = px.scatter(
        region_data, x='Willing (%)', y='AwarenessOpportunity', text='Region', size='Total',
        hover_name='Region', hover_data={'Region': False, 'Willing (%)': ':.1f%', 'AvgAwareness': ':.3f', 'AwarenessOpportunity': ':.1f', 'Total': True},
        labels={
            'Willing (%)': f'Market Attractiveness (% WTP @ ${wtp_price_point})',
            'AwarenessOpportunity': 'Market Entry Opportunity (Higher = Lower Awareness)',
            'Total': 'Respondents'
            },
        size_max=50
    )
    fig.add_hline(y=median_opportunity, line_dash="dash", line_color="grey", annotation_text="Median Opportunity", annotation_position="bottom right")
    fig.add_vline(x=median_wtp, line_dash="dash", line_color="grey", annotation_text="Median WTP", annotation_position="top left")

    anno_opts = dict(showarrow=False, font=dict(size=10, color='grey'), bgcolor="rgba(255,255,255,0.7)")
    fig.add_annotation(x=region_data['Willing (%)'].max(), y=region_data['AwarenessOpportunity'].max(), text="High Priority", align='right', valign='top', xanchor='right', yanchor='top', **anno_opts)
    fig.add_annotation(x=region_data['Willing (%)'].max(), y=region_data['AwarenessOpportunity'].min(), text="Optimize", align='right', valign='bottom', xanchor='right', yanchor='bottom', **anno_opts)
    fig.add_annotation(x=region_data['Willing (%)'].min(), y=region_data['AwarenessOpportunity'].max(), text="Monitor", align='left', valign='top', xanchor='left', yanchor='top', **anno_opts)
    fig.add_annotation(x=region_data['Willing (%)'].min(), y=region_data['AwarenessOpportunity'].min(), text="Lower Priority", align='left', valign='bottom', xanchor='left', yanchor='bottom', **anno_opts)

    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    fig.update_layout(
        title={'text': matrix_title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
        plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif", font_color=XM_COLORS['text'],
        xaxis_ticksuffix="%", yaxis_ticksuffix=" ",
        xaxis_range=[min(0, region_data['Willing (%)'].min()*0.9), max(1, region_data['Willing (%)'].max()*1.05)], # X-axis range unchanged
        # INCREASE Y-AXIS PADDING MORE:
        yaxis_range=[min(0, region_data['AwarenessOpportunity'].min()*0.85), max(1, region_data['AwarenessOpportunity'].max()*1.25)], # Changed min padding to 0.85, max padding to 1.25
        margin=dict(t=65, b=5, l=40, r=40), # Keep current margins
        height=600 # <--- ADDED EXPLICIT FIGURE HEIGHT (adjust value if needed)
    )
    return fig

# 4. Price Sensitivity Curves (Van Westendorp) - FINAL REVISION -> INVERTED STYLE
# Implements inverted VW curves, requested labels/colors, and shaded region
def create_vw_chart(segment_filter='All Segments', group_filter=None): # Removed df argument
    """Creates an inverted Van Westendorp chart with user-specified labels and shaded acceptable range."""
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Van Westendorp chart", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty: # Check if the globally loaded data (now local df) is empty
        return go.Figure().update_layout(title_text="No data available for Van Westendorp chart", plot_bgcolor='white', paper_bgcolor='white')

    # --- Filtering logic remains the same ---
    if segment_filter != 'All Segments':
        if 'Segment' not in df.columns:
            return go.Figure().update_layout(title_text="Error: 'Segment' column missing", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df[df['Segment'] == segment_filter].copy()
        if df_filtered.empty:
            return go.Figure().update_layout(title_text=f"No VW data for segment: {segment_filter}", plot_bgcolor='white', paper_bgcolor='white')
        segment_title_part = f" ({segment_filter})"
    else:
        df_filtered = df.copy()
        segment_title_part = " (All Segments)"

    group_title_suffix = ""
    if group_filter == 'Control':
        control_col = 'IsControlGroup'
        if control_col not in df_filtered.columns:
            print("Error: 'IsControlGroup' column missing for Control group filter.")
            return go.Figure().update_layout(title_text="Error: Missing 'IsControlGroup' column", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df_filtered[df_filtered[control_col] == 1].copy()
        group_title_suffix = " - Control Group"
    elif group_filter == 'Test':
        test_col = 'SawFeatureExplanation'
        if test_col not in df_filtered.columns:
            print("Error: 'SawFeatureExplanation' column missing for Test group filter.")
            return go.Figure().update_layout(title_text="Error: Missing 'SawFeatureExplanation' column", plot_bgcolor='white', paper_bgcolor='white')
        df_filtered = df_filtered[(df_filtered[test_col] == 1)].copy()
        group_title_suffix = " - Test Group"

    chart_title = f"Van Westendorp Price Sensitivity{group_title_suffix}{segment_title_part}"

    # --- Data validation and price range calculation (remains the same) ---
    vw_cols = ['VW_TooCheap', 'VW_Bargain', 'VW_Expensive', 'VW_TooExpensive']
    if not all(col in df_filtered.columns for col in vw_cols):
        print("Error: Missing one or more Van Westendorp columns for the selected filters.")
        return go.Figure().update_layout(title_text="Error: Missing VW data columns", plot_bgcolor='white', paper_bgcolor='white')

    for col in vw_cols:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    df_filtered = df_filtered.dropna(subset=vw_cols)

    n_respondents = len(df_filtered)
    if n_respondents < 10:
        return go.Figure().update_layout(title_text=f"Insufficient VW data for filter (N={n_respondents})", plot_bgcolor='white', paper_bgcolor='white')

    try:
        min_price_data = df_filtered[vw_cols].min().min()
        max_price_data = df_filtered[vw_cols].max().max()
        if pd.isna(min_price_data) or pd.isna(max_price_data) or min_price_data >= max_price_data:
             return go.Figure().update_layout(title_text=f"Invalid price range in VW data for filter", plot_bgcolor='white', paper_bgcolor='white')
        min_price = max(0, min_price_data * 0.8) # Ensure non-negative start
        max_price = max_price_data * 1.1
        if max_price <= min_price: max_price = min_price + 50
        price_range = np.linspace(min_price, max_price, num=100)
    except Exception as e:
        print(f"Error calculating price range for VW: {e}")
        return go.Figure().update_layout(title_text=f"Error calculating price range: {e}", plot_bgcolor='white', paper_bgcolor='white')

    # --- Calculate INVERTED cumulative percentages ---
    # Descending: % cumulative *ABOVE* threshold
    perc_too_inexpensive = [(df_filtered['VW_TooCheap'] > p).mean() * 100 for p in price_range] # % saying price > P is Too Cheap (Desc)
    perc_inexpensive = [(df_filtered['VW_Bargain'] > p).mean() * 100 for p in price_range]      # % saying price > P is Bargain (Desc)

    # Ascending: % cumulative *BELOW* threshold
    perc_expensive = [(df_filtered['VW_Expensive'] < p).mean() * 100 for p in price_range]        # % saying price < P is Expensive (Asc)
    perc_too_expensive = [(df_filtered['VW_TooExpensive'] < p).mean() * 100 for p in price_range] # % saying price < P is Too Expensive (Asc)

    fig = go.Figure()

    # --- Add traces using INVERTED curves and USER labels/colors ---
    # Order based on example image (Red, Purple, Blue, Green)
    fig.add_trace(go.Scatter(x=price_range, y=perc_too_expensive, mode='lines', name='Too Expensive', line=dict(color=VW_CHART_COLORS_INVERTED['Too Expensive'])))
    fig.add_trace(go.Scatter(x=price_range, y=perc_expensive, mode='lines', name='Expensive', line=dict(color=VW_CHART_COLORS_INVERTED['Expensive'])))
    fig.add_trace(go.Scatter(x=price_range, y=perc_inexpensive, mode='lines', name='Inexpensive', line=dict(color=VW_CHART_COLORS_INVERTED['Inexpensive'])))
    fig.add_trace(go.Scatter(x=price_range, y=perc_too_inexpensive, mode='lines', name='Too Inexpensive', line=dict(color=VW_CHART_COLORS_INVERTED['Too Inexpensive'])))

    # --- Intersection calculation logic (same function as before) ---
    def find_intersection(x, y1, y2):
        diff = np.array(y1) - np.array(y2)
        sign_change = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            if idx + 1 >= len(x): return None, None
            x1, x2 = x[idx], x[idx+1]
            y11, y12 = y1[idx], y1[idx+1]
            y21, y22 = y2[idx], y2[idx+1]
            try:
                denominator = (y12 - y11) - (y22 - y21)
                if abs(denominator) < 1e-6: return None, None
                intersect_x = x1 + (y21 - y11) * (x2 - x1) / denominator
                if x2 - x1 != 0:
                    intersect_y = y11 + (y12 - y11) * (intersect_x - x1) / (x2 - x1)
                else:
                    intersect_y = (y11 + y21) / 2
                if intersect_x >= min(x1, x2) and intersect_x <= max(x1, x2):
                    return intersect_x, intersect_y
                else:
                    return None, None
            except (ZeroDivisionError, IndexError, FloatingPointError):
                 return None, None
        return None, None

    # --- Calculate INVERTED intersection points ---
    # PMC: Expensive (Asc) vs Too Inexpensive (Desc)
    pmc_x, pmc_y = find_intersection(price_range, perc_expensive, perc_too_inexpensive)
    # PME: Too Expensive (Asc) vs Inexpensive (Desc)
    pme_x, pme_y = find_intersection(price_range, perc_too_expensive, perc_inexpensive)
    # OPP: Too Expensive (Asc) vs Too Inexpensive (Desc) - NOTE: This is sometimes IDP in other conventions
    opp_x, opp_y = find_intersection(price_range, perc_too_expensive, perc_too_inexpensive)
    # IDP (If needed): Expensive (Asc) vs Inexpensive (Desc)
    idp_x, idp_y = find_intersection(price_range, perc_expensive, perc_inexpensive)


    intersection_points = {'PMC': (pmc_x, pmc_y), 'PME': (pme_x, pme_y), 'OPP': (opp_x, opp_y)}
    # Optionally add IDP if you want to display it:
    # intersection_points['IDP'] = (idp_x, idp_y)

    # --- Add annotations and vertical lines ---
    pmc_val, pme_val = None, None # Store values for shaded region
    for name, (x_val, y_val) in intersection_points.items():
        if x_val is not None and x_val >= min_price and x_val <= max_price:
            # Store PMC/PME for shading
            if name == 'PMC': pmc_val = x_val
            if name == 'PME': pme_val = x_val

            # Add dashed line ONLY for PMC and PME
            if name in ['PMC', 'PME']:
                 fig.add_vline(x=x_val, line_dash="dash", line_color="grey") # Dashed lines for PMC/PME

            # Add annotation for all points
            y_pos = 50
            if y_val is not None: y_pos = max(5, min(95, y_val))
            ay_offset = -40 if y_pos > 60 else 40
            # Minor adjustments for specific points if needed based on typical overlaps
            if name == 'OPP' and y_pos < 40: ay_offset = 40
            if name == 'PMC' and y_pos < 40: ay_offset = 40
            if name == 'PME' and y_pos > 60: ay_offset = -40

            fig.add_annotation(
                x=x_val, y=y_pos,
                text=f"{name}<br>${x_val:.0f}", showarrow=True, arrowhead=1,
                bgcolor="rgba(255,255,255,0.7)", borderpad=2,
                ax=0, ay=ay_offset
            )

    # --- Add shaded region between PMC and PME ---
    if pmc_val is not None and pme_val is not None:
        # Ensure shading happens between the lower and higher value regardless of order
        lower_bound = min(pmc_val, pme_val)
        upper_bound = max(pmc_val, pme_val)
        fig.add_vrect(
            x0=lower_bound, x1=upper_bound,
            fillcolor="grey", opacity=0.15, layer="below", line_width=0,
            annotation_text="Acceptable Range", annotation_position="top left",
            annotation=dict(font_size=10, font_color='grey')
        )

    # --- Layout update ---
    fig.update_layout(
        title={'text': chart_title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
        xaxis_title="Price Point ($)", yaxis_title="Percentage of Respondents (%)",
        yaxis_ticksuffix="%", yaxis_range=[0, 105], xaxis_tickprefix="$",
        plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif", font_color=XM_COLORS['text'],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# 5. NEW: Regression Coefficient Plot (Code unchanged from previous corrected version)
def create_regression_coef_plot(): # Removed df argument
    """Runs OLS regression and creates a coefficient plot."""
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Regression Analysis", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty: # Check if the globally loaded data (now local df) is empty
        return go.Figure().update_layout(title_text="No data for Regression Analysis", plot_bgcolor='white', paper_bgcolor='white')

    try:
        # --- Data Preparation ---
        saw_feat_exp_col = 'SawFeatureExplanation'
        if saw_feat_exp_col not in df.columns:
             print("Error: 'SawFeatureExplanation' column missing for regression filter.")
             return go.Figure().update_layout(title_text="Error: Missing 'SawFeatureExplanation' column", plot_bgcolor='white', paper_bgcolor='white')

        df_reg = df[df[saw_feat_exp_col] == 1].copy()

        # Define outcome and predictors (using cleaned names)
        outcome = 'VW_Expensive'
        categoricals = ['IncomeBracket', 'AgeGroup', 'Region']
        continuous = ['BrandAwarenessProxy_DistWt', 'ValuePerceptionPost']
        predictors = categoricals + continuous

        required_cols = [outcome] + predictors
        if not all(col in df_reg.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_reg.columns]
            print(f"Warning: Missing columns for regression: {missing}")
            return go.Figure().update_layout(title_text=f"Missing columns for Regression: {missing}", plot_bgcolor='white', paper_bgcolor='white')

        df_reg = df_reg[required_cols].dropna()

        if len(df_reg) < 50:
             return go.Figure().update_layout(title_text=f"Insufficient data for Regression (N={len(df_reg)} after filtering)", plot_bgcolor='white', paper_bgcolor='white')

        # --- Preprocessing ---
        scaler = StandardScaler()
        df_reg[continuous] = scaler.fit_transform(df_reg[continuous])
        for col in categoricals:
            df_reg[col] = df_reg[col].astype('category')

        formula = f"{outcome} ~ "
        formula += " + ".join([f"C({col})" for col in categoricals])
        formula += " + " + " + ".join(continuous)
        print(f"Regression Formula: {formula}")

        model = smf.ols(formula, data=df_reg).fit()
        print(model.summary())

        params = model.params.reset_index()
        conf_int = model.conf_int().reset_index()
        params.columns = ['Variable', 'Coefficient']
        conf_int.columns = ['Variable', 'Conf_Int_Lower', 'Conf_Int_Upper']
        plot_data = pd.merge(params, conf_int, on='Variable')
        plot_data = plot_data[plot_data['Variable'] != 'Intercept']

        plot_data['Display_Variable'] = plot_data['Variable'].str.replace(r'C\((.*?)\)\[T\.(.*?)\]', r'\1: \2', regex=True)
        plot_data['Display_Variable'] = plot_data['Display_Variable'].str.replace(r'C\((.*?)\)', r'\1', regex=True)
        plot_data['Display_Variable'] = plot_data['Display_Variable'].str.replace('_', ' ', regex=False)
        plot_data = plot_data.sort_values(by='Coefficient', key=abs, ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data['Coefficient'],
            y=plot_data['Display_Variable'],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Coefficient',
            hoverinfo='text',
            hovertext=[f"{row['Display_Variable']}<br>Coef: {row['Coefficient']:.2f}<br>CI: [{row['Conf_Int_Lower']:.2f}, {row['Conf_Int_Upper']:.2f}]"
                       for index, row in plot_data.iterrows()]
        ))
        fig.add_trace(go.Scatter(
            x=plot_data['Conf_Int_Lower'],
            y=plot_data['Display_Variable'],
            mode='markers', marker=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=plot_data['Conf_Int_Upper'],
            y=plot_data['Display_Variable'],
            mode='markers', marker=dict(color='rgba(0,0,0,0)'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=plot_data['Conf_Int_Upper'] - plot_data['Coefficient'],
                arrayminus=plot_data['Coefficient'] - plot_data['Conf_Int_Lower'],
                visible=True,
                thickness=1,
                width=0,
                color='grey'
            ),
            showlegend=False, hoverinfo='none'
        ))
        fig.add_vline(x=0, line=dict(color='red', width=1, dash='dash'))

        chart_height = 350 + len(plot_data) * 25
        fig.update_layout(
            title='Key Statistical Drivers of WTP (VW Expensive)',
            xaxis_title='Coefficient (Impact on VW Expensive Price)',
            yaxis_title='Predictor Variable',
            yaxis=dict(tickmode='array', tickvals=plot_data['Display_Variable'], ticktext=plot_data['Display_Variable']),
            plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif",
            font_color=XM_COLORS['text'],
            margin=dict(l=200, r=30, t=50, b=50),
            height=chart_height,
            showlegend=False
        )
        return fig

    except Exception as e:
        print(f"Error running regression or creating plot: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().update_layout(title_text=f"Error during Regression Analysis: {e}", plot_bgcolor='white', paper_bgcolor='white')


# 7. Top 3 Drivers Chart (Cross-Segment) - Renumbered
def create_top_drivers_chart(): # Removed df argument
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Top Drivers Chart", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty or not all(c in df.columns for c in ['Segment', 'Top3Driver1', 'Top3Driver2', 'Top3Driver3']):
        return go.Figure().update_layout(title_text="Insufficient data for Top Drivers Chart", plot_bgcolor='white', paper_bgcolor='white')
    try:
        df_melt = df.melt(
            id_vars=['RespondentID', 'Segment'],
            value_vars=['Top3Driver1', 'Top3Driver2', 'Top3Driver3'],
            var_name='DriverRank',
            value_name='Driver'
        )
        df_melt = df_melt.dropna(subset=['Driver', 'Segment'])
        if df_melt.empty:
             return go.Figure().update_layout(title_text="No valid driver data found", plot_bgcolor='white', paper_bgcolor='white')
        driver_counts = df_melt.groupby(['Segment', 'Driver']).size().reset_index(name='Mentions')
        segment_totals = df.groupby('Segment')['RespondentID'].count().reset_index(name='TotalRespondents')
        segment_totals['TotalMentions'] = segment_totals['TotalRespondents'] * 3
        driver_perc = pd.merge(driver_counts, segment_totals, on='Segment')
        driver_perc['Percentage'] = (driver_perc['Mentions'] / driver_perc['TotalMentions'] * 100).round(1)
        overall_importance = driver_perc.groupby('Driver')['Percentage'].mean().sort_values(ascending=False)
        driver_perc['Driver'] = pd.Categorical(driver_perc['Driver'], categories=overall_importance.index, ordered=True)
        color_map = {seg: XM_COLORS.get(seg, '#CCCCCC') for seg in driver_perc['Segment'].unique()}
        fig = px.bar(
            driver_perc.sort_values(['Driver', 'Segment']),
            y='Driver', x='Percentage', color='Segment', barmode='group', orientation='h',
            labels={'Percentage': '% Mentioned in Top 3', 'Driver': 'Purchase Driver', 'Segment': 'Customer Segment'},
            title="Key Purchase Drivers by Segment (% Mentioned in Top 3)",
            text='Percentage', color_discrete_map=color_map
        )
        fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        num_drivers = len(overall_importance.index)
        chart_height = max(400, 50 + num_drivers * 85)
        fig.update_layout(
            yaxis={'categoryorder':'array', 'categoryarray': overall_importance.index},
            xaxis_ticksuffix="%", plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif",
            font_color=XM_COLORS['text'], hovermode="y unified",
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, title=None),
            title={'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
            margin=dict(l=120, r=20, t=50, b=80, pad=5), height=chart_height
        )
        return fig
    except Exception as e:
        print(f"Error creating Top Drivers chart: {e}")
        return go.Figure().update_layout(title_text=f"Error generating Top Drivers chart: {e}", plot_bgcolor='white', paper_bgcolor='white')


# 8. Primary Usage Chart (Cross-Segment) - Renumbered
def create_primary_usage_chart(): # Removed df argument
    if app_data is None: # Check global
        return go.Figure().update_layout(title_text="Data unavailable for Primary Usage Chart", plot_bgcolor='white', paper_bgcolor='white')
    df = app_data # Use global data

    if df.empty or not all(c in df.columns for c in ['Segment', 'PrimaryUsage']):
        return go.Figure().update_layout(title_text="Insufficient data for Primary Usage Chart", plot_bgcolor='white', paper_bgcolor='white')
    try:
        df_usage = df.dropna(subset=['Segment', 'PrimaryUsage'])
        if df_usage.empty:
            return go.Figure().update_layout(title_text="No valid usage data found", plot_bgcolor='white', paper_bgcolor='white')
        usage_counts = df_usage.groupby(['Segment', 'PrimaryUsage']).size().reset_index(name='Count')
        usage_perc = usage_counts.groupby('Segment').apply(lambda x: x.assign(Percentage=(x['Count'] / x['Count'].sum() * 100))).reset_index(drop=True)
        segment_order = sorted(usage_perc['Segment'].unique(), reverse=True)
        usage_order = ['Everyday casual wear', 'Work/Office (casual environment)', 'Travel', 'Specific activity needing convenience', 'Style/Fashion statement', 'Other']
        usage_perc['Segment'] = pd.Categorical(usage_perc['Segment'], categories=segment_order, ordered=True)
        usage_perc['PrimaryUsage'] = pd.Categorical(usage_perc['PrimaryUsage'], categories=[u for u in usage_order if u in usage_perc['PrimaryUsage'].unique()], ordered=True)
        usage_perc = usage_perc.sort_values(['Segment', 'PrimaryUsage'])
        color_sequence = px.colors.qualitative.Pastel
        fig = px.bar(
            usage_perc, y='Segment', x='Percentage', color='PrimaryUsage', orientation='h',
            labels={'Percentage': 'Percentage of Segment (%)', 'PrimaryUsage': 'Primary Usage', 'Segment': 'Customer Segment'},
            title="Primary Usage Scenario by Segment", text_auto='.0f', color_discrete_sequence=color_sequence
        )
        fig.update_traces(texttemplate='%{x:.0f}%', insidetextanchor='middle')
        fig.update_layout(
            xaxis_ticksuffix="%", xaxis_range=[0, 100],
            yaxis={'categoryorder':'array', 'categoryarray': segment_order},
            plot_bgcolor='white', paper_bgcolor='white', font_family="Inter, sans-serif",
            font_color=XM_COLORS['text'], hovermode="y unified",
            legend=dict(traceorder='normal', orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title=None),
            title={'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18}},
            bargap=0.15, margin=dict(b=100)
        )
        return fig
    except Exception as e:
        print(f"Error creating Primary Usage chart: {e}")
        return go.Figure().update_layout(title_text=f"Error generating Primary Usage chart: {e}", plot_bgcolor='white', paper_bgcolor='white')

print("Defining app layout...")
# --- App Layout ---
# Layout structure remains the same, only the VW chart function call changes behavior
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # dcc.Store(id='dummy-store'), # Removed dummy-store
    dcc.Store(id='selected-values-store'),
    dcc.Store(id='limitation-content-store'), # <-- NEW Store
    html.Div(id='dummy-output', style={'display': 'none'}),

    # --- Sidebar ---
    # (Sidebar code remains unchanged from original)
    html.Div([
        dbc.Card(dbc.CardBody([html.Div([html.Span("Project Sections", className="section-header")],className="d-flex align-items-center")], className="p-2"), className="mb-2 section-header-card"),
        dbc.Nav([
            dbc.NavLink(html.Span(["Project Overview & Hypothesis", html.I(className="fas fa-chevron-down nav-arrow")], className="nav-link-content"), href="#overview", id="nav-overview", active="exact", className="nav-link parent-nav-link", n_clicks=0),
            dbc.Collapse(html.Div([ dbc.NavLink("Executive Summary", href="#executive-summary", id="nav-executive-summary", active="exact", className="nav-link nested-link"), dbc.NavLink("Goal", href="#goal", id="nav-goal", active="exact", className="nav-link nested-link"), dbc.NavLink("Objective", href="#objective", id="nav-objective", active="exact", className="nav-link nested-link"), dbc.NavLink("Observations & Hypotheses", href="#observations", id="nav-observations", active="exact", className="nav-link nested-link"), dbc.NavLink("Approach", href="#approach", id="nav-approach", active="exact", className="nav-link nested-link"), ], className="nested-links-container"), id="collapse-overview", is_open=False),
            dbc.NavLink(html.Span(["Methodology & Design", html.I(className="fas fa-chevron-down nav-arrow")], className="nav-link-content"), href="#methodology-design", id="nav-methodology-design", active="exact", className="nav-link parent-nav-link", n_clicks=0),
            dbc.Collapse(html.Div([ dbc.NavLink("Two-Phase Gabor-Granger", href="#gabor-granger", id="nav-gabor-granger", active="exact", className="nav-link nested-link"), dbc.NavLink("Assuming Uplift in WTP", href="#wtp", id="nav-wtp", active="exact", className="nav-link nested-link"), dbc.NavLink("Van Westendorp", href="#van-westendorp", id="nav-van-westendorp", active="exact", className="nav-link nested-link"), dbc.NavLink("Control Group", href="#control-group", id="nav-control-group", active="exact", className="nav-link nested-link"), dbc.NavLink("Competitive Adjustment", href="#competitive-adjustment", id="nav-competitive-adjustment", active="exact", className="nav-link nested-link"), dbc.NavLink("Population Sampling", href="#population-sampling", id="nav-population-sampling", active="exact", className="nav-link nested-link"), dbc.NavLink("Distance-Weighted Awareness Proxy", href="#awareness-proxy", id="nav-awareness-proxy", active="exact", className="nav-link nested-link"), ], className="nested-links-container"), id="collapse-methodology", is_open=False),
            dbc.NavLink(html.Span(["My Role: Guiding AI", html.I(className="fas fa-chevron-down nav-arrow")], className="nav-link-content"), href="#my-role", id="nav-my-role", active="exact", className="nav-link parent-nav-link", n_clicks=0),
            dbc.Collapse(html.Div([ dbc.NavLink("Rejecting Inappropriate Methodology", href="#rejecting-methodology", id="nav-rejecting-methodology", active="exact", className="nav-link nested-link"), dbc.NavLink("Identifying Critical Measurement Ambiguity", href="#identifying-ambiguity", id="nav-identifying-ambiguity", active="exact", className="nav-link nested-link"), dbc.NavLink("Refining Awareness Proxy Methods", href="#refining-proxy", id="nav-refining-proxy", active="exact", className="nav-link nested-link"), dbc.NavLink("Pushing for Deeper Portfolio Analysis", href="#pushing-for-deeper-portfolio-analysis", id="nav-pushing-for-deeper-portfolio-analysis", active="exact", className="nav-link nested-link"), ], className="nested-links-container"), id="collapse-my-role", is_open=False),
            dbc.NavLink("Survey Flowchart", href="#survey-flow", id="nav-survey-flow", active="exact", className="nav-link"),
            # Updated Interactive Visualizations Link
            dbc.NavLink(html.Span(["Interactive Visualizations", html.I(className="fas fa-chevron-down nav-arrow")], className="nav-link-content"), href="#visualizations", id="nav-visualizations", active="exact", className="nav-link parent-nav-link", n_clicks=0),
            # New Collapse for Visualizations
            dbc.Collapse(html.Div([
                # MODIFIED: Point existing VW link to the first (control) chart container
                dbc.NavLink("Van Westendorp", href="#vw-chart-control-container", id="nav-vw-chart", active="exact", className="nav-link nested-link"),
                # (Optionally add a second link for Test chart here if desired, but keeping it simple as per instructions)
                dbc.NavLink("Willingness to Pay: Pre vs Post Feature Explanation", href="#wtp-gg-chart-container", id="nav-wtp-gg-chart", active="exact", className="nav-link nested-link"),
                dbc.NavLink("Regional WTP", href="#regional-map-container", id="nav-regional-map", active="exact", className="nav-link nested-link"),
                # Note: Expansion Matrix not included as per user request
                dbc.NavLink("Key Statistical Drivers of WTP", href="#regression-coef-container", id="nav-regression-coef", active="exact", className="nav-link nested-link"), # Shortened name slightly
                dbc.NavLink("Key Purchase Drivers by Segment", href="#top-drivers-container", id="nav-top-drivers", active="exact", className="nav-link nested-link"),
                dbc.NavLink("Primary Usage Scenario by Segment", href="#primary-usage-container", id="nav-primary-usage", active="exact", className="nav-link nested-link"),
            ], className="nested-links-container"), id="collapse-visualizations", is_open=False),
            # End New Collapse
            dbc.NavLink("Segment Profiles", href="#segment-profiles", id="nav-segment-profiles", active="exact", className="nav-link"),
            # --- NEW Sections from previous step ---
            dbc.NavLink("Key Findings", href="#key-findings", id="nav-key-findings", active="exact", className="nav-link"),
            # --- ADDED Missing NavLinks ---
            dbc.NavLink("Business Impact", href="#business-impact", id="nav-business-impact", active="exact", className="nav-link"),
            dbc.NavLink("Limitations", href="#limitations", id="nav-limitations", active="exact", className="nav-link"), # Link to Limitations section
            dbc.NavLink("Data and Sources", href="#data", id="nav-data", active="exact", className="nav-link"),
            dbc.NavLink("Technology Stack", href="#tech-stack", id="nav-tech-stack", active="exact", className="nav-link"),
            # --- END Added Missing NavLinks ---
        ], vertical=True, pills=True),
    ], style=SIDEBAR_STYLE, className="dashboard-sidebar"),

    # --- Main content ---
    html.Div([
        html.H1(["Kizik Footwear: ", html.Br(), "Simulated Price Sensitivity & Expansion Strategy"], className="text-center mb-4", id="title", style={'margin-top': '2rem', 'font-size': '2.5rem', 'font-weight': '700'}),

        # --- Text Sections ---
        html.Div([
            html.H3("Project Overview & Hypotheses", className="mb-2", style={'font-size': '3rem', 'font-weight': '600'}),
            dcc.Markdown("""
            """, className="text-content")
        ], id="overview", className="text-container"),

# All of the following components are now children of this single parent Div.
# This makes them behave as a single, cohesive block, just like your "Challenge and Observation" example.
html.Div([
    # Section 1: Executive Summary
    html.H3("Executive Summary", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
    dcc.Markdown("""
    This project demonstrates an end-to-end market research process, from hypothesis to strategic business recommendations. Guiding AI, I developed a parameter-informed simulation in Python that analyzed the impact of feature education and brand awareness on willingness to pay.
    """, className="text-content"),

    # Section 2: Key Findings
    # The title and its paragraph are now siblings at the same level.
    html.P([html.Strong("Key Findings", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
    dcc.Markdown('''
    Feature education directly impacts revenue potential. Explaining the technology boosted the price "Premium Enthusiasts" were willing to pay by $25 (a roughly 20% increase), lifting their WTP from $121 to $146.
    
    Additionally, brand awareness *does* increase willingness to pay, but not as much as other factors such as region or income.
    ''', className="text-content", style={'margin-top': '0'}),

    # Section 3: Key Business Recommendation
    html.P([html.Strong("Key Business Recommendation", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
    dcc.Markdown('''
    At least in this simulation, there is a significant market gap for an "Explore Lite" or trail-focused shoe priced in the $125 - $140 range. This could capture demand from many underserved segments currently priced out of Kizik's higher-tier models. 
    ''', className="text-content", style={'margin-top': '0'}),

    # Section 4: Skills Demonstrated
    html.P([html.Strong("Skills Demonstrated", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
    dcc.Markdown('''
    * **Technical solution design:** Architected and built a full-stack data application using Python, AI models, and Dash to investigate a complex business problem from beginning to end.
    * **Proactive problem-solving & diagnostic acumen:** Identified pricing discrepancies in secondhand markets, formulated data-driven hypothesis, and designed a simulation to diagnose the root cause.
    * **Data-driven recommendations & strategic insight:** Translated complex simulation outputs into clear, value-added business recommendations tailored to specific customer segments' needs.
    * **Complex data translation for non-technical audiences:** Created detailed project breakdown that articulates advanced statistical methodologies and their business implications for a fairly general audience.
    ''', className="text-content", style={'margin-top': '0'}),

# The id and className are applied to the parent Div that contains everything.
], id="executive-summary", className="text-container"),

        html.Div([
            html.H3("Goal", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
            
            My goal with this project was to explore out-of-state price sensitivity to Kizik shoes relative to brand awareness and feature awareness, using a parameter-informed simulation.

            Why a simulation? It was a great opportunity to tie in to current trends in market research (rapid and growing adoption of synthetic data), while also allowing me to quickly build a survey to test my hypotheses.
            
            """, className="text-content")
        ], id="goal", className="text-container"),
        html.Div([
            html.H3("Objective", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
                To construct a realistic, statistically plausible dataset grounded in publicly available data - without having direct access to private data to build a GAN to analyze, refine, and ultimately generate models to create true synthetic responses. This modeled dataset allowed for rigorous testing of advanced survey methodologies, including:

                * Gabor-Granger
                * Van Westendorp
                * Multiple Linear Regression

                ...validation of analytical workflows (segmentation, regression), and the generation of preliminary insights regarding Kizik's pricing and market expansion opportunities, essentially demonstrating an end-to-end research process.
            """, className="text-content")
        ], id="objective", className="text-container"),

        html.Div([
            html.H3("Observations & Hypotheses", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
                What sparked this project was an interesting trend I was seeing: I had purchased four pairs of Kiziks over the last few months - all in 9/10 condition or brand new, never worn - *but they were all significantly underpriced* relative to their retail value.

                I had purchased them at around 60-70% off, and they were all shipped from locations outside of Utah.

                Given that Kiziks are more or less a household name in Utah, it had me wondering: **was the pricing so low because of low out-of-state brand awareness?** Kizik's a new company, but not that new. **And, more generally, would people be willing to pay more for Kiziks if they knew just how easy it was to slip them on?**

                I knew I didn't have the time or resources to conduct an extensive field survey, but from the 2025 Market Research Trends report, podcasts with Isabelle Zdatny and Ali Henriques, I knew that synthetic data - or something adjacent to it - just might be able to get me some answers.
            """, className="text-content")
        ], id="observations", className="text-container", style={'margin-bottom': '0rem !important'}),

        html.Div([
            html.Img(
                src=app.get_asset_url('kizik pricing.png'),
                alt="discounts from secondhand markets",
                 style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
            )
        ], id="kizik-pricing", style={ 
                            'width': '80%',     # Or '95%', '1000px', '1200px' etc.
                            'max-width': '1200px', # Optional: A new, larger max-width if desired
                            'margin-left': 'auto', 
                            'margin-right': 'auto',
                            'margin-top': '0.01rem',   # Add some spacing above
                            'margin-bottom': '4rem' # Add some spacing below
                           }),

        html.Div([
            html.H3("Approach and Process", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
                Utilizing Python, AI models, and publicly available data (US Census, BLS, Kizik store locations), I was able to curate a parameter-informed simulation to model consumer behavior and test these hypotheses without the delays of traditional surveys.

                * A parameter-informed simulation is a simulation where the inputs (parameters) are values informed by real-world data.

                **Below is a high-level overview of my overall process and the tools I used:**
            """, className="text-content")
            
        ], id="approach", className="text-container", style={'margin-top': '1.5rem !important', 'margin-bottom': '0rem !important'}),

        html.Div([
            html.Img(
                src=app.get_asset_url('qualtrics process.png'),
                alt="red bull babyyyyyy",
                 style={'display': 'block', 'margin': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
            )
        
        ], id="qualtrics-process", style={ 
                            'width': '80%',     # Or '95%', '1000px', '1200px' etc.
                            'max-width': '1200px', # Optional: A new, larger max-width if desired
                            'margin-left': 'auto', 
                            'margin-right': 'auto',
                            'margin-top': '0.01rem',   # Add some spacing above
                            'margin-bottom': '1rem' # Add some spacing below
                           }),

        html.Div([ # Standard text container for the block
            html.P([html.Strong("", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown("""
                **More detail in the Tech Stack section below, but here's a rough outline of the process I used to craft this:**
                * Read & mark up every page of the three 2025 Trends Reports: Market, Consumer, and Employee Experience
                * Listen to podcast episodes with Qualtrics thought leaders Isabelle Zdatny and Ali Henriques
                * Observe that synthetic data usage is driving massive investments in the face of a feedback recession
                * *So why NOT get familiar with the survey tools Product Experts and other employees discuss with clients daily?*
                * Wrestle with Claude and Gemini to craft a parameter-informed simulation that gets as close as I can to "creating" synthetic data
                * Use Cursor and Windsurf to write the code for this site
                * Learn a LOT, fine-tune prompting skills after hundreds of iterations across multiple features
                
                """, className="text-content", style={'margin-top': '0'}),

        ], id="approach-explanation", className="text-container", style={'margin-top': '1.5rem !important', 'margin-bottom': '0rem !important'}),

        html.Div([
            html.H3("Methodology and Design", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""The following methodologies were employed to ensure robust data collection and analysis:""", className="text-content")
        ], id="methodology-design", className="text-container"),

html.Div([
    html.H3("Two-Phase Gabor-Granger", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),

    # The text before the link remains in a Markdown component
    dcc.Markdown("""
        Including Gabor-Granger was instrumental in this study, as it allowed us to assess willingness to pay (WTP, or purchase likelihood) before and after a specific intervention in the study (Sections 4 and 6, respectively).

        This intervention, in section 5 - had Test Group participants being shown a demonstration of the hands-free technology, while Control Group participants were not shown it. As mentioned in the Explanation column in Section 6, the Control Group served to establish a comparison baseline to determine the effects of the feature explanation, allowing me to test my hypothesis that willingness to pay *could likely* increase with a brief feature explanation. In this case, it was purely a "conceptual" explanation given the limitations of our simulation. Below, I discuss artificially creating an uplift in WTP to represent the effects of the feature explanation.
    """, className="text-content"),

    # The paragraph with the link is now constructed with html components
    html.P([
        # Wrap the link and colon in Strong and Em for bold/italic styling
        html.Strong(html.Em([
            html.A(
                "Definition", 
                href="https://www.qualtrics.com/marketplace/gabor-granger-pricing-sensitivity-study/", 
                target="_blank" # This makes it open in a new tab
            ),
            ":"
        ])),
        # Wrap the rest of the text in Em for italic styling
        html.Em(
            " a pricing technique used to determine the relationship between pricing and demand (willingness to pay). This relationship is expressed through a price elasticity curve, which we’ve modeled in the interactive chart titled: “Willingness to Pay: Pre vs. Post Feature Explanation (All Segments).” This curve is generated by asking respondents how much they’d be willing to pay at each price point and also determines the revenue-maximizing point for a given product or service."
        )
    ], className="text-content") # Use the same class for consistent styling

], id="gabor-granger", className="text-container"),

html.Div([
    html.H3("Assuming Uplift in WTP", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),

    # Keep the parts that are pure Markdown in their own components
    dcc.Markdown("""
        **Determining WTP uplift upfront instead of modeling it**
        
        You'd think that because it's a simulation, it would be able to precisely model - to some extent - if people would be willing to pay more (measured by WTP) for Kiziks after the intervention (the feature explanation mentioned above). But that's precisely the core limitation that makes simulations, well... simulations!

        We can build the survey, assign the right weights, population sampling, etc., but what we can't do is have it model the true psychological impact of new information - of somebody actually trying the shoes on. We'd need real people to produce that *new*, raw, experiential data. Instead, what we do here is simulate the *result* of the value of that impact or perspective shift, instead of trying to model the complex, cognitive process of genuine value perception pre- and post-try-on (not a real phrase but work with me here). As it's a parameter-informed simulation, a 15% WTP average uplift was chosen as a conservative but plausible input *parameter* for the purposes of this experiment.

        **Why choose 15%?**
    """, className="text-content"),

    # --- THIS IS THE KEY CHANGE ---
    # Construct the paragraph with links using a list of components inside an html.P
    html.P([
        "Identifying the specific WTP for Kizik's HandsFree tech would require extensive, in-field market research. However, the 15% average uplift used in this study aligns well with general findings (",
        html.A("Lab42", href="https://www.dexigner.com/news/28046", target="_blank"),
        ", ",
        html.A("Ketchum Innovation", href="https://www.amic.media/media/files/file_352_807.pdf", target="_blank"),
        ", ",
        html.A("Circana/NYU Stern CSB", href="https://www.stern.nyu.edu/sites/default/files/2023-04/FINAL%202022%20CSB%20Report%20for%20website.pdf", target="_blank"),
        ") from pricing strategy best-practice principles, conjoint analysis studies across consumer goods, and general benchmarks for convenience/performance-oriented product features. Given that range, 15% felt like a conservative yet meaningful lift for a competitive product category like footwear."
    ], className="text-content"),
    # --- END OF KEY CHANGE ---

    # The rest of the text in another Markdown component
    dcc.Markdown("""
        So choosing 15% represents a plausible but conservative estimate that reflects the genuine innovation that is Kizik's HandsFree tech. It acknowledges that the legitimate convenience and accessibility benefits it offers are significant enough to merit a reasonable premium over standard shoes relative to other traditional, valued shoe features. It also takes into account the fact that we're using a purely conceptual explanation rather than a real, physical trial. Additionally, it allows for the effects to be decently significant in the simulation's outputs, providing a reasonable basis for testing my core hypothesis that post-try-on (or feature education) increases WTP. I would have loved to have some kind of sentiment analysis based on a text-based description of the hands-free tech, but mapping text-based sentiment scores to a correlating purchase behavior across multiple product features, segments (and their correlating factors) was outside the scope of this project. The solitary uplift we assigned for WTP was sufficient enough for the purposes of this experiment.
    """, className="text-content")

        ], id="wtp", className="text-container"),

html.Div([
    html.H3("Van Westendorp Price Sensitivity Meter", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
    
    # The first paragraph remains in a Markdown component
    dcc.Markdown("""
        Including the Van Westendorp Price Sensitivity Meter was crucial in helping us understand acceptable price ranges. In this experiment, it also served as a complement to our Gabor-Granger method, which uses pre-set prices. Additionally, it aided in helping calibrate our willingness to pay data, especially at lower price ranges.
    """, className="text-content"),

    # The paragraph with the link is now an html.P component
    html.P([
        # Wrap the link and colon in Strong and Em for bold/italic styling
        html.Strong(html.Em([
            html.A(
                "Definition",
                href="https://www.qualtrics.com/marketplace/vanwesterndorp-pricing-sensitivity-study/",
                target="_blank"  # This makes it open in a new tab
            ),
            ":"
        ])),
        # Wrap the rest of the text in Em for italic styling
        html.Em(
            " A kind of price perception survey that helps identify a range of acceptable prices that fit well within customer price perceptions and expectations. It asks four price points: Too Cheap, Bargain, Expensive, Too Expensive."
        )
    ], className="text-content")  # Apply the same class for consistent styling

], id="van-westendorp", className="text-container"),
        html.Div([
            html.H3("Control Group", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
                While our Test Group was being assessed on purchase likelihood before and after a text-based conceptual feature explanation, our Control Group served as a baseline to understand the true effects of that feature explanation (again, in Section 5, the Control group is *not* shown the text-based conceptual feature explanation).
            """, className="text-content")
        ], id="control-group", className="text-container"),
        
html.Div([
    # --- Title Section ---
    html.H3("Competitive Adjustment", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),

    # --- Paragraph with embedded links ---
    # This paragraph is constructed as a list of components (strings and html.A for links)
    # to allow for hyperlinking specific phrases within the text.
    html.P([
        "To take into account the competitive nature of the market, a competitive adjustment range of 15% - 25% was suggested by Claude, with me settling on a mid-range 20%. It's well established that consumers are more likely to pay less when competitive options are present, \"trading down\" for a better-perceived mix of value and price (",
        html.A("McKinsey", href="https://www.mckinsey.com/industries/consumer-packaged-goods/our-insights/the-state-of-the-us-consumer-2024", target="_blank"),
        "). This applies well to the footwear industry, with 78% of shoppers walking away from a footwear purchase due to cost (",
        html.A("Alix Partners", href="https://www.alixpartners.com/newsroom/press-release-new-consumer-study-shows-78-of-shoppers-have-walked-away-from-a-footwear-purchase-due-to-cost/", target="_blank"),
        "), ideally to find lower-cost alternatives. Additionally, 70% of consumers would halt their purchase, \"opting to compare prices elsewhere or wait for discounts\" (",
        html.A("Simon-Kucher & Partners", href="https://www.simon-kucher.com/en/insights/footwear-consumer-priorities-industry-insights", target="_blank"),
        "). This last data point reinforces the logic to apply a competitive handicap: if a 5% increase in price cuases 70% of consumers to abandon a purchase, it shows that WTP for select models is extremely variable when competitive options are present. While it's hard to put a finger on an exact number for this handicap, 20% in this experiment serves as a representation of the effect competitive options have on WTP - at least for this limited experiment. Specific handicaps would require extensive field testing."
    ], className="text-content"),

    # --- Second paragraph using dcc.Markdown for standard text ---
    dcc.Markdown("""
I chose to add this in after asking about the most glaring errors in the survey, and Claude mentioned that not including it would likely overestimate willingness to pay, given that 1) the survey design at that time asked specifically about Kizik without comparison to alternatives and 2) that real purchase decisions almost always tend to involve trade-offs between brands. This is a uniform competitive handicap that applies to all segments.
    """, className="text-content")

], id="competitive-adjustment", className="text-container"),

html.Div([
    html.H3("Distance-Weighted Awareness Proxy", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
    
    # The text before the links remains in a Markdown component
    dcc.Markdown("""
        Without access to Kizik's internal data on brand awareness - and driven by the limitations in our parameter-informed simulation, this proxy served as a derived metric, simulating brand awareness based on store locations, giving more weight to closer stores (distance decay). Simple store density wasn't enough, so this distance-weighted awareness proxy was as close as I could get without access to local ad data or data on store/age duration in market.

        I made additional refinements (see "Refining Awareness Proxy Methods" under the "My Role: Guiding AI" section) that distinguished different brand awareness levels between Kizik-owned flagship stores and retail stores, given that branded flagship stores will contribute to a higher overall brand awareness than a traditional retailer selling dozens of shoe brands.
    """, className="text-content"),

    # The paragraph with the links is now an html.P component
    html.P([
        "Well-established marketing research shows a strong correlation between physical retail presence and brand awareness (",
        html.A(
            "ICSC",
            href="https://admiralrealestate.com/icsc-report-the-halo-effect-how-bricks-impact-clicks/#:~:text=The%20study%20followed%20the%20effects,see%20reductions%20in%20online%20sales.",
            target="_blank"  # Opens in a new tab
        ),
        " and ",
        html.A(
            "Brookfield Properties",
            href="https://www.brookfieldproperties.com/en/our-businesses/retail/insights.html",
            target="_blank"  # Opens in a new tab
        ),
        ")."
    ], className="text-content") # Apply the same class for consistent styling

], id="awareness-proxy", className="text-container"),

        # --- My Role Sections ---
        # (Keep original My Role sections)
        html.Div([
            html.H3("My Role: Guiding AI", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""
            
            My role in this project was not to passively use AI, but to actively guide and direct it as a strategic partner. This involved assessing its outputs, rejecting flawed methodologies, and refining its logic to ensure the final simulation was both realistic and analytically sound.

            I used a variety of different AI tools to help guide the survey design process, from Claude Pro to Gemini 2.5 Pro. More details on the tools themselves can be found in the "Tech Stack" section, but AI played a crucial role in helping refine the survey design and generate the code that powers this site and the dashboards.
            
            More importantly, it proved to be a quick way to flesh my ideas out. Just like with any other technology, though, it can make mistakes. 
            
            My role here was to assess its output and refine the survey design, effectiveness and appropriateness of methodological tools, and measure the overall questions against survey best-practices, rejecting or accepting proposed changes and suggesting a more appropriate alternative when needed.
            
            Below are some of the better examples of the way I was able to guide AI as part of this process.

            *Please note: any typos or errors in my prompts are either due to typing too fast or from the voice to text software I was using.*
            """, className="text-content")
        ], id="my-role", className="text-container", style={'width': '100%'}),
        html.Div([
            html.H3("Rejecting Inappropriate Methodology", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown('''
                I determined that conjoint analysis, a powerful tool suggested by Claude, would not be appropriate for this specific experiment due to the unique limitations of my simulation.
            ''', className="text-content", style={'margin-top': '0', 'margin-bottom': '1rem'}),
            html.Div([
                html.Img(
                    src=app.get_asset_url('conjoint_rejection_discussion.png'),
                    alt="Discussion rejecting conjoint analysis for this scenario",
                    style={'display': 'block', 'margin': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
                )
            ], className="chart-card", style={'padding': '1rem', 'width': '49%', 'margin': '1rem auto'}),
            html.P([html.Strong("Challenge and Observation", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                    While assessing various tools and survey methodologies, Claude suggested conjoint analysis. Although conjoint works great in so many other surveys, including it here would have been incorrect and imprecise. Like other survey tools, it does have its limitations, and given the nature of our parameter-informed simulation, we would have effectively been generating imprecise responses due to the fact that the responses would have been formed from an interpretation (emphasis on interpretation!) of a description of a physical test, and not the tactile, experience-based knowledge gained from a real respondent taking that test (fit, feel, ease of use, etc.).
            ''', className="text-content", style={'margin-top': '0'}),

            html.P([html.Strong("My Guidance", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                    I rejected this approach before it was adopted into our design, pointing out its lack of fit due to the various kinds of tradeoffs a respondent would need to make (which wouldn't have worked in our simulated scenario).

                    *A note on that as well: there's also the matter of how appropriate it would be to even include Kiziks in an analysis like these: while conjoint analyses are almost exclusively digital and my rejection on its inclusion here was partially on the premise of needing a physical comparison rather than a simulated one, one of the core issues (as I pointed out) is that we'd be asking them to make comparisons between something with relatively well-known features (standard slip-on shoes) and something with truly "innovative," new features (like Kizik's HandsFree tech). Conjoint works great when you're comparing products with different features, features that most people have some universal level understanding of. Never end sentences with prepositions, btw. Unless you're knee-deep in a conjoint swamp.

                    So conjoint is doubly ineffective and inappropriate across those dimensions.
            ''', className="text-content", style={'margin-top': '0'}),

            html.P([html.Strong("Outcome", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                    We steered clear of an inappropriate methodology and opted instead to focus on more appropriate tools like the two-phase Gabor-Granger and Van Westendorp, making sure that the survey focused on techniques that would suitably accommodate the uniqueness of the product and the simulated nature of the responses.
            ''', className="text-content", style={'margin-top': '0'})
        ], id="rejecting-methodology", className="text-container"),

        html.Div([
            html.H3("Identifying Critical Measurement Ambiguity", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""Ambiguity is fine in some surveys and unacceptable in others. Here, I had the opportunity to direct Claude to make an important distinction regarding a core question that had been created that needed additional clarity. In its previous ambiguous state, it would have marred our survey data.""", className="text-content"),
            html.Div([
                html.Img(
                    src=app.get_asset_url('ex 3.2 measurement ambiguity.png'),
                    alt="Further discussion on measurement ambiguity",
                     style={'display': 'block', 'margin': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
                )
            ], className="chart-card", style={'padding': '1rem', 'width': '49%', 'margin': '1rem auto'}),
            html.Div([
                html.Img(
                    src=app.get_asset_url('ex 3.1 - measurement ambiguity.png'),
                    alt="Identifying measurement ambiguity in the survey",
                     style={'display': 'block', 'margin': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
                )
            ], className="chart-card", style={'padding': '1rem', 'width': '49%', 'margin': '1rem auto'}),

            # --- START: Inserted Text Block ---
            html.P([html.Strong("Challenge and Observation")], style={'font-size': '18px', 'font-weight': 'bold', 'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''While the AI had been very useful in helping craft survey questions, it sometimes tended to overlook how the wording might skew results. I noticed that the verbiage in one of the questions was too vague, and could lead to some issues with the responses it generated. Specifically, Question 3.1 didn’t describe a clear enough distinction between what was already available on the market and what Kizik had to offer. “Hands-free” tech sounds just like another pair of loafers or Vans or Crocs, or in and out garden variety errand shoe we slip on to do chores, and if we’re trying to create a clear distinction between the two to show any kind of meaningful correlation with other core variables in our hypothesis (like willingness to pay for new features!), descriptions like these would essentially flatten that distinction at best in a simulation and, at worst, confuse real respondents in the field.''', className="text-content", style={'margin-top': '0'}),

            html.P([html.Strong("My Guidance")], style={'font-size': '18px', 'font-weight': 'bold', 'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''I had to explicitly lay out the fundamental flaws - clearly outlining how somebody could get confused, and why the way that it was described was essentially the same thing as a slip-on shoe.''', className="text-content", style={'margin-top': '0'}),

            html.P([html.Strong("Outcome")], style={'font-size': '18px', 'font-weight': 'bold', 'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''ID’ing this ambiguity and the subsequent direction I gave to use specific but non-leading language direct from Kizik’s site led to important revisions in the survey.''', className="text-content", style={'margin-top': '0'}),
            # --- END: Inserted Text Block ---
        ], id="identifying-ambiguity", className="text-container"),
        html.Div([
            html.H3("Refining Awareness Proxy Methods", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""
I directed Claude to explore better awareness proxies given the nature of our survey, and further refined the metric it provided by assessing its strengths and selecting recommendations to improve it.
            """, className="text-content"),
             html.Div([
                dbc.Carousel(
                    items=[
                        {"src": app.get_asset_url("2.1 TRUE proxy first one - anything close to awareness question.png"), "caption": "Trying to find a good proxy for brand awareness", "captionClassName": "d-none"},
                        {"src": app.get_asset_url("2.2 TRUE proxy - how defensible question.png"), "caption": "Evaluating proposed proxies", "captionClassName": "d-none"},
                        {"src": app.get_asset_url("ex 2.3 TRUE - awareness proxy.png"), "caption": "Determining defensibility of store proximity as a proxy", "captionClassName": "d-none"},
                        {"src": app.get_asset_url("ex 2.4 TRUE - awareness proxy.png"), "caption": "Exploring ways to strengthen store proximity", "captionClassName": "d-none"},
                        {"src": app.get_asset_url("2.5 TRUE - awareness proxy.png"), "caption": "Adding distance weighting", "captionClassName": "d-none"},
                    ],
                    controls=True, indicators=True, interval=None, slide=True, id="awareness-proxy-carousel",
                    style={"--bs-carousel-control-icon-bg": "transparent"}
                ),
                html.Div(id="external-carousel-caption", style={'text-align': 'center'})
            ], className="chart-card", id="carousel-card", style={'padding': '1rem', 'width': '85%', 'margin': '1.5rem auto 0 auto'}),
            html.P([html.Strong("Challenge and Observation", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown('''
                    Claude's initial build of the simulation had used store count per capita to serve as a proxy for brand awareness. Given the simulated nature of our responses, I was concerned our proxy wouldn't be enough, and Claude confirmed my suspicions indicating that store count per capita would only yield a 7/10 defensibility, given that it was essentially ignoring store proximity to population centers (a key part in any store's pre-planning phase).
                ''', className="text-content", style={'margin-top': '0'}),
            html.P([html.Strong("My Guidance", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown('''
                    I'm always wary of AI models being suspiciously too agreeable, so I challenged this weakness and directly asked for improvement: "Is there anything that we can do to bring that seven closer to a 10?" Claude suggested incorporating distance weighting which I then explicitly directed it to include. I additionally asked for an updated defensibility score, which had increased to 8 - 8.5/10.
                ''', className="text-content", style={'margin-top': '0'}),
html.P([html.Strong("Outcome", style={'font-size': '18px', 'font-weight': 'bold', 'font-family': 'Arial, sans-serif'})], style={'margin': '1.5rem 0 0.5rem 0'}),
dcc.Markdown('''
    Pushing for improvement and directing that improvement resulted in Claude implementing a distance-weighted awareness proxy (calculate_weighted_awareness function) in the code used to run the simulation. The model now takes into account the decreased impact of stores farther away from population centers, creating a more analytically sound and realistic awareness metric, which was then used in regression and segmentation analyses later on in the experiment.
''', className="text-content", style={'margin-top': '0', 'font-family': 'Arial, sans-serif', 'font-size': '16px', 'margin-bottom': '2rem'}),

html.P([html.Strong("Additional Awareness Logic Refinements: Flagship Stores vs Retail Store Weights", style={'font-size': '18px', 'font-weight': 'bold', 'font-family': 'Arial, sans-serif'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                After making this distinction with Claude, I later directed a key change to the way the stores were weighted in this newly refined awareness logic.

                When I built a list of all 372 stores offering Kiziks, I had to make the distinction to Gemini 2.5 Pro Experimental (AI model by Google used for additional post-Claude analysis) that 6 of them were flagship stores (Kizik-owned and branded retail stores selling exclusively Kizik products), and the rest were 3rd party retail stores.

                Making this distinction was important, as areas with Kizik flagship stores would obviously have a higher brand awareness (and higher variation in WTP) than just a Payless or a mom-and-pop shoe store that *happened* to sell Kiziks along with the dozens of other brands on their shelves.

                At this request, Gemini introduced the following changes that created a more robust simulated model of how retail presence influences brand awareness, moreso than just simple store counts per state.

                Without getting too technical, the final *BrandAwarenessProxy_DistWt* metric incorporates several factors.

                **Refined Brand Awareness Proxy (Distance-Weighted with Flagship Distinction):**

                1. **Flagship Store Bonus:** States with one of the 6 Kizik-owned flagship stores receive a significant, baseline awareness bonus, reflecting the higher visibility and brand impact of dedicated retail spaces. States directly adjacent to flagship states receive a smaller "spillover" bonus. Utah receives an additional small bonus as the home state.
                2. **Third-Party Retailer Density:** The density of third-party retailers (stores per 100k population) contributes to awareness, but at a lower base weight than flagship stores, reflecting their potentially less prominent branding or product placement.
                3. **Distance Weighting (Implicit via Density Scaling):** The impact of third-party stores is scaled non-linearly using `np.log1p(stores_per_100k * 5)`. This simulates a distance-weighting effect, where the marginal awareness gain from adding more stores diminishes (i.e., going from 0 to 1 store per 100k has more impact than going from 10 to 11). It assumes higher density correlates with closer average proximity to population centers.
                4. **Random Variation:** A small amount of random noise is added to reflect other unmodeled factors.
                5. **Capping:** The final proxy score is capped between 0.01 and 0.50 to keep it within a plausible range.

                * *Factors list summarized by Gemini*

                Pointing out the heightened awareness in areas with flagship stores and lowered awareness in areas with retailers that just *happened* to sell Kizik shoes allowed the simulation to be much more defensible than just using simple store density as the initial brand awareness metric. Without additional internal brand data, including these refinements allow for more refined simulated measurements of price sensitivity.
            ''', className="text-content", style={'margin-top': '0', 'font-family': 'Arial, sans-serif', 'font-size': '16px'})
            ], id="refining-proxy", className="text-container"),


# --- New addition - start here ---
        html.Div([
            html.H3("Pushing for Deeper Portfolio Analysis", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown('''
                As part of analyzing the business impact of Kizik's pricing strategy, I noticed some interesting trends among the amount and type of models offered at certain pricing tiers, and instructed Gemini to uncover them.
            ''', className="text-content", style={'margin-top': '0', 'margin-bottom': '1rem'}),
            
            html.Div([
                html.Img(
                    src=app.get_asset_url('guiding gemini.png'),
                    alt="Pushing for deeper portfolio analysis",
                    style={'display': 'block', 'margin': 'auto', 'max-width': '100%', 'height': 'auto', 'border-radius': '8px'}
                )
            ], className="chart-card", style={'padding': '1rem', 'width': '49%', 'margin': '1rem auto'}),
            html.P([html.Strong("Observation", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                What I had observed from Kizik's site was that price itself was fairly category-defining, with the the most rugged, outdoor-oriented "Explore" models priced at $149 and above, effectively pricing out segments whose global max WTP ceiling was at $140. Obviously, the ceiling is going to be lower in this simulation due to the parameters being used and the fact that this simulation can't capture true value perception like a real respondent can through social media, trying them on for themselves, word of mouth, etc.
                
                This pricing tier/lack of models discrepancy analysis could be performed even without this sim, but I was able to take a closer look at how it was affecting the quantity of models available to my "respondents."
            ''', className="text-content", style={'margin-top': '0'}),
            html.P([html.Strong("My Guidance", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
                After scraping the model list (women's only) from Kizik's site, adding Kizik-defined category tags such as "Relax," "Active," and "Explore," I pushed Gemini to identify if "there's potential for inclusion of x amount of models at y category for z segment," essentially uncovering missed opportunities, that - at least in my simulation - could appear to benefit underserved segments who might desire the features but are effectively limited by their WTP ceiling.
            ''', className="text-content", style={'margin-top': '0'}),

            html.P([html.Strong("Outcome", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown('''
            My findings here are outlined in more detail in the Business Impact section, but in short, I discovered that there was decent opportunity to introduce more "lite" versions of the "Explore" models, effectively meeting segment demand for those features/preferences at lower prices.
            
            It's likely that there's current demand for this, as many brands use this same paring-down strategy of higher-priced models to address needs of segments with lower WTP. 
            ''', className="text-content", style={'margin-top': '0'}),

        ], id="pushing-for-deeper-portfolio-analysis", className="text-container"),

        # --- End My Role Sections ---

# --- Survey Flow Section - MODIFIED for Wider Display ---
        html.Div([ # Outer Div for the entire section (this one is now full-width)
            
            # NEW WRAPPER DIV FOR THE TITLE (to center it and give it max-width)
            html.Div([
                html.H3(
                    "Survey Flowchart", 
                    className="mb-2", 
                    style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'} # H3 takes full width of this new parent
                )
            ], style={ # Style this wrapper like your other centered title wrappers or text-container
                'max-width': '780px',  # Or '900px' if you used that for the Viz title wrapper
                'margin-left': 'auto',
                'margin-right': 'auto',
                'width': '100%' 
            }),
            # END NEW WRAPPER DIV

            # Inner Div wrapping ONLY the image (this one can remain wider)
            html.Div([ 
                html.Img( 
                    src=app.get_asset_url('final flowchart.png'), 
                    alt="Qualtrics Survey Flowchart", 
                    style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'max-width': '100%', 'height': 'auto'} 
                )
            ], style={'width': '95%', 'margin': '1.5rem auto 0 auto'}),

        ], id="survey-flow", style={'width': '100%', 'margin-bottom': '3rem'}),

        # --- Visualizations Section (REORDERED) ---
        html.Div([ # This is the main Div with id="visualizations"
            
            html.Div([ 
                html.H3(
                    "Interactive Visualizations", 
                    className="mb-2", 
                    style={
                        'font-size': '1.5rem', 
                        'font-weight': '600', 
                        'width': '100%', 
                        'margin-bottom': '1.5rem' 
                    }
                )
            ], style={ 
                'max-width': '900px',      
                'margin-left': 'auto',   
                'margin-right': 'auto',    
                'width': '100%',           
                'padding-left': '0px', # Adjusted to align with centered content
                'padding-right': '0px'      
            }),
            
            # --- Van Westendorp Control Chart ---
            html.Div(dcc.Graph(id='van-westendorp-chart-control', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     id='vw-chart-control-container'), 
            # MODIFIED TOOLBAR STRUCTURE
            html.Div(className="static-chart-toolbar", children=[
                html.Div(className="toolbar-controls-wrapper", children=[
                    # Segment Group ONLY
                    html.Div(className="toolbar-control-group segment-group", children=[
                        dbc.Label("Customer Segment", className="toolbar-label",),
                        html.Div(id={'type': 'static-segment-options', 'chart': 'vw-control'}, className='toolbar-options-list',)
                    ])
                ])
            ]),
            
            # --- Van Westendorp Test Chart ---
            html.Div(dcc.Graph(id='van-westendorp-chart-test', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     id='vw-chart-test-container'),
            # MODIFIED TOOLBAR STRUCTURE
            html.Div(className="static-chart-toolbar", children=[
                html.Div(className="toolbar-controls-wrapper", children=[
                    # Segment Group ONLY
                    html.Div(className="toolbar-control-group segment-group", children=[
                        dbc.Label("Customer Segment", className="toolbar-label"),
                        html.Div(id={'type': 'static-segment-options', 'chart': 'vw-test'}, className='toolbar-options-list')
                    ])
                ])
            ]),

           # --- Overview and Analysis Text Block for VW Charts ---
            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
The Van Westendorp price sensitivity meter helps us understand how our overall price is perceived. It reveals how much our segments are generally willing to pay for these shoes and identifies the acceptable range of prices based on the data. The chart illustrates what percentage of respondents per segment are willing to pay relative to any given price.

In our study, it complements the Gabor-Granger method (which uses pre-set prices) and helps calibrate WTP, especially at the lower price boundary.


To determine prices in an unbiased way, four questions are asked:

- **Too Expensive**: "At what price would you consider these hands-free casual comfort shoes to be so expensive that you would not consider buying them?"
- **Expensive**: "At what price would you consider these shoes to be starting to get expensive, but you would still consider buying them?"
- **Inexpensive or Bargain**: "At what price would you consider these shoes to be a bargain - a great buy for the money?"
- **Too Cheap or Too Inexpensive**: "At what price would you consider these shoes to be priced so low that you would feel the quality couldn't be very good?"


The relationship between these prices and their answers forms two critical price points:

- **PMC (Point of Marginal Cheapness)**: Represents the lower bound of acceptable price ranges.
- **PME (Point of Marginal Expensiveness)**: Represents the upper bound of acceptable price ranges.

The prices between PMC and PME form the **overall acceptable range of price points**. The **OPP (Optimal Price Point)** is the specific price where the percentage of respondents who find the price "too cheap" equals the percentage who find it "too expensive." This range is where the majority of respondents in a particular segment would consider purchasing the product.
                """, className="text-content", style={'margin-top': '0'})
            ], className="text-container", style={'margin-bottom': '2rem'}),

            # --- WTP Gabor-Granger Chart ---
            html.Div(dcc.Graph(id='wtp-gg-chart', config=STANDARD_GRAPH_CONFIG),
                     className="chart-card",
                     style={'margin-top': '4rem'},
                     id='wtp-gg-chart-container'),
            # MODIFIED TOOLBAR STRUCTURE
            html.Div(className="static-chart-toolbar", children=[
                html.Div(className="toolbar-controls-wrapper", children=[
                    # Segment Group ONLY
                    html.Div(className="toolbar-control-group segment-group", children=[
                        dbc.Label("Customer Segment", className="toolbar-label"),
                        html.Div(id={'type': 'static-segment-options', 'chart': 'wtp-gg'}, className='toolbar-options-list')
                    ])
                ])
            ]),
            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    This shows the willingness to pay at certain price points, and illustrates the impact of the feature explanation, essentially helping answer one of the core questions in my hypothesis: “Are people willing to pay more *after* receiving an explanation of how the tech works?”
                """, className="text-content", style={'margin-top': '0'}),
                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
Yes! While some segments are more flexible on price, the overall trend is that each segment saw a post-explanation lift in willingness to pay, suggesting that explaining the tech does matter for price.
                """, className="text-content", style={'margin-top': '0'})
            ], className="text-container"),


            # --- Regional WTP Map ---
            html.Div(dcc.Graph(id='regional-wtp-map', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     style={'margin-top': '4rem'},
                     id='regional-map-container'),
            # MODIFIED TOOLBAR STRUCTURE (WITH BOTH CONTROLS)
            html.Div(className="static-chart-toolbar", children=[
                html.Div(className="toolbar-controls-wrapper", children=[
                    # Segment Group
                    html.Div(className="toolbar-control-group segment-group", children=[
                        dbc.Label("Customer Segment", className="toolbar-label"),
                        html.Div(id={'type': 'static-segment-options', 'chart': 'regional-map'}, className='toolbar-options-list')
                    ]),
                    # Divider
                    html.Div(className="toolbar-divider"),
                    # WTP Group
                    html.Div(className="toolbar-control-group wtp-group", children=[
                        dbc.Label("WTP Price Point", className="toolbar-label"),
                        html.Div(id={'type': 'static-wtp-options', 'chart': 'regional-map'}, className='toolbar-options-grid')
                    ])
                ])
            ]),
            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    This chart shows per-state WTP based on the post-explanation probability data (simulated percentage likelihood of purchase) averaged across all respondents (both Control and Test groups combined) who fall within the selected segment filter.
                """, className="text-content", style={'margin-top': '0'}),
                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    The 'West' region has an estimated VW_Expensive that is $20.87 higher than the baseline region, the Midwest. Being in the West, Northeast, or South region significantly predicts higher WTP compared to the baseline.
                """, className="text-content", style={'margin-top': '0'})
            ], className="text-container"),

            # --- Regression Coefficient Plot (No Controls) ---
            html.Div(dcc.Graph(id='regression-coef-plot', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     style={'margin-top': '4rem'},
                     id='regression-coef-container'),
            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""

*   This is a **Regression Coefficient Plot**.
*   It visually summarizes the results of a statistical model (specifically, Ordinary Least Squares - OLS regression) that tried to predict the price point respondents considered "**Expensive**" based on the Van Westendorp question (`VW_Expensive` column in your data).
*   The goal is to identify which characteristics (demographics, attitudes, awareness) have a **statistically significant** relationship with how high a price someone will tolerate before deeming it "Expensive". And of course, this only represents the Test Group (those who did see the conceptual feature explanation of Kizik's hands-free tech). Running the MLR only on the Test Group allows us to understand what influences the perception of "Expensive" on those who are *aware* of the technology. Ideally, by doing this, you'd be able to start building more accurate segments for marketing campaigns, product research, etc.



**What the Components Mean**



1.  **Y-Axis (Vertical):** Lists the **Predictor Variables** included in the regression model. These are the factors the model tested to see if they influence the "Expensive" price threshold.
    *   Examples: `IncomeBracket: Less than $25k`, `AgeGroup: 65+`, `Region: West`, `ValuePerceptionPost`, `BrandAwarenessProxy_DistWt`.
    *   Note: Categorical variables (like Income, Age, Region) are shown relative to a baseline category that isn't displayed (e.g., the effect of being "65+" is compared to the youngest age group, which is the baseline). Continuous variables (`ValuePerceptionPost`, `BrandAwarenessProxy_DistWt`) were likely standardized, meaning their effect is per standard deviation change.
2.  **X-Axis (Horizontal):** Shows the **Coefficient** value. This number represents the estimated **strength and direction** of the relationship between that predictor variable and the `VW_Expensive` price point.
3.  **Blue Dots:** The estimated **coefficient value** for each predictor.
4.  **Horizontal Grey Lines (Error Bars):** The **95% Confidence Interval** for each coefficient. This shows the range where the *true* effect likely lies. If this line crosses the vertical zero line, the effect is generally **not statistically significant** (p > 0.05).
5.  **Vertical Dashed Red Line:** Represents a coefficient of **zero (0)**. This is the line of "no effect."


**How to Interpret It**

*   **Right of the Red Line (Positive Coefficient):** Variables with dots to the right are associated with respondents naming a **higher** price as "Expensive." For example, if `Region: West` is significantly to the right, it suggests people in the West tend to tolerate higher prices before calling them expensive compared to the baseline region.
*   **Left of the Red Line (Negative Coefficient):** Variables with dots to the left are associated with respondents naming a **lower** price as "Expensive." For example, `IncomeBracket: Less than $25k` being significantly to the left suggests this group considers *lower* prices to be "Expensive" compared to the baseline income group.
*   **Statistical Significance:** If a variable's horizontal grey line **does not cross** the vertical red line, its effect is statistically significant. We are reasonably confident this factor has a real association with the "Expensive" threshold. If the line **does cross** zero, we cannot confidently say it has an effect distinct from random chance.
*   **Magnitude:** The further a dot is from the red zero line (in either direction), the stronger the estimated impact of that variable is on the "Expensive" price threshold.


**In Simple Terms**

This chart tells you **which factors significantly push the "Expensive" price point up or down** for the respondents in your (simulated) data. It helps identify the characteristics of people who are more or less price-sensitive, specifically regarding when they start thinking a price is getting high (but might still consider buying).
* *Technical Summary by Gemini*
                """, className="text-content", style={'margin-top': '0'}),
            ], className="text-container"),

            html.Div([ 
                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    People in the West, Northeast, and those in higher income brackets ($150k+) are more likely to tolerate higher prices before labeling them "Expensive." Conversely, lower income brackets produce the opposite effect, while the tail ends of the age distribution seem to be at odds with each other relative to price perception: the older you are (65+) the less likely you are to consider higher prices while younger age brackets (25-34) have a higher tolerance for raised prices before labeling them "Expensive." 
                    
                    In part, I believe that's why this experiment resonates with me so much: those who most likely need these shoes the most (due to mobility issues) are the ones who are most price sensitive. Generally speaking, it's part of their generation's sociohistorical profile: they tend to focus more on practicality and are therefore more likely to be more price-sensitive. That's why effective marketing campaigns are so necessary: well-communicated innovations solving real pain points can genuinely alleviate legitimate, painful struggle.
                """, className="text-content", style={'margin-top': '0'}),
            ], className="text-container"),

            # --- Top 3 Drivers Chart (No Controls) ---
            html.Div(dcc.Graph(id='top-drivers-chart', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     style={'margin-top': '4rem'},
                     id='top-drivers-container'),
            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    This chart is the result of asking our "respondents'" to rate the importance of a variety of different shoe attributes, like comfort, price, style, hands-free convenience, etc. It's a bar chart representation of a cross-tabs (cross-tabulations) analysis. Cross-tab analysis shows the number or frequency of respondents that share the same value for a given attribute, like the percentage of "Curious Skeptics" who value Style.


                """, className="text-content", style={'margin-top': '0'}),
            ], className="text-container"),

            html.Div([ 
                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    Charts like these are important because price isn't always a core motivator when making purchase decisions, and it varies by segment. Some segments are willing to trade it off for comfort or the unique  tech that Kizik offers ("HandsFree Convenience"). For marketers and ad teams, it helps define core marketing messages by understanding which benefits - and tradeoffs - resonate most with consumers
                    As you can see, Premium Enthusiasts greatly value Style and Kizik's tech, while Traditionalists are exactly as they sound with an overwhelming emphasis on Price, Durability, and Comfort, while averaging 4.5% across the other purchase drivers.
                """, className="text-content", style={'margin-top': '0'}),
            ], className="text-container"),

            # --- Primary Usage Chart (No Controls) ---
            html.Div(dcc.Graph(id='primary-usage-chart', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     style={'margin-top': '4rem'},
                     id='primary-usage-container'),
            html.Div([
                html.P([
                    html.Strong("What it is: "),
                    "This asks the respondent to select the main situation or context where they would envision themselves wearing the described hands-free shoes."
                ]),
                html.P([
                    html.Strong("Answer Options: "),
                    "'Everyday casual wear', 'Work/Office (casual environment)', 'Travel', 'Specific activity needing convenience (e.g., carrying items, mobility challenges)', 'Style/Fashion statement', 'Other'."
                ]),
                html.P([
                    html.Strong("Why it's useful: "),
                    "It adds context to ",
                    html.Em("why"),
                    " someone might value the shoe and its features. Someone buying for 'Everyday casual wear' might prioritize comfort and style differently than someone buying specifically for 'Mobility challenges'(example scenario under ‘Special activity needing convenience’) who might heavily prioritize the hands-free convenience. It helps the researcher understand the job-to-be-done, in addition to potentially informing marketing and product development."
                ]),
            ],
            className="text-container", 
            style={'margin-top': '0rem', 'margin-bottom': '2rem'} 
            ),
            # --- End Chart Order ---

        ], id="visualizations", style={'width': '100%'}),

        # --- NEW: Interactive Segment Information Display ---
        html.Div([
            html.H3("Segment Profiles", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),
            dcc.Markdown("""
            These segments represent the simulated, pre-defined consumer groups present in the study - while the data is helpful, I've gone ahead and included what that data could represent in terms of a hypothetical persona for each segment. It's fun to imagine them as real people, with real lives, likes, and dislikes.
            """, className="text-content") # Updated placeholder
        ], id="segment-profiles", className="text-container", style={'margin-bottom': '0rem !important'}),

        html.Div([
             # NEW: Title Display Area (populated by callback)
            html.H4(
                SEGMENT_INFO[SEGMENT_ORDER[0]]["title"], # Load default title
                id='segment-title-display',
                className="segment-info-title", # Reuse class for consistency
                style={'font-weight': 'bold', 'margin-bottom': '1rem', 'margin-top': '1rem !important', 'padding-left': '1rem'} # Add padding
            ),
            # Content Display Area (populated by callback)
            html.Div(id='segment-content-display', children=create_segment_content(SEGMENT_ORDER[0])), # Load default content
            # Segment Selector Tabs
            html.Div(id='segment-selector-tabs', className='segment-selector-tabs', children=[
                html.Button( # Changed from html.Div to html.Button
                    segment,
                    id={'type': 'segment-tab', 'index': segment},
                    className=f"segment-tab-button {'active' if i == 0 else ''}", # Use new class, keep active logic for now
                    n_clicks=0
                ) for i, segment in enumerate(SEGMENT_ORDER)
            ])
        ], id="segment-info-container", className="segment-info-container"), # Use DIFFERENT class than chart-card
        # --- END: Interactive Segment Information Display ---

        # --- Key Findings Section MODIFIED ---
        html.Div([
            html.H3("Key Findings", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            html.P([html.Strong("Outcome of my two core hypotheses:", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    Before we advance to the rest of the key findings, I'd like to take a moment to address the two core hypotheses I set out to test:
                    
                    ***"Would respondents' WTP increase if they knew just how easy it was to slip them on?"***

                    ***"Is out-of-state WTP so low because of low out-of-state brand awareness?"***

                    It's been so interesting to see how these hypotheses played out in the data, and even more fun to be able to replicate what it could be like to draw conclusions from a survey with hard data to back it up (albeit limited because of the nature of our experiment).

                    """, 
                    className="text-content", style={'margin-top': '0', 'margin-bottom': '4rem !important'}),
        
            # --- Keep the original Overview block that was here ---
            html.Div([ # Standard text container for the block
                html.P([html.Strong("Would respondents' WTP increase if they knew just how easy it was to slip them on?", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '4rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    """, 
                    className="text-content", style={'margin-top': '0'}),

            # --- INSERTED TITLE and TABLE ---
                html.P("WTP Analysis Summary by Segment", style={'text-align': 'center', 'font-weight': 'bold', 'margin-top': 'rem', 'margin-bottom': '0.5rem'}),
                dcc.Markdown("""

                    ```text
                    -------------------------------------------------------------------------------------
                    WTP Pre/Post ($) and Median Lift by Segment
                    (WTP $ = Interpolated Price where 50% are Willing; Pre uses Control, Post uses Test)
                    (Lift %/pp uses Test Group Initial vs Post across $119, $139, $159)
                    -------------------------------------------------------------------------------------
                            Segment       Pre-WTP     Post-WTP   Median Abs Lift   Median % Lift
                    ------------------------- ---------- ----------- ----------------- --------------
                          Curious Skeptics      $112       $127         15.1 pp          43.6%
                    Interested Pragmatists      $117       $133         15.5 pp          46.7%
                       Premium Enthusiasts      $121       $146         19.5 pp          48.6%
                           Traditionalists       $97        $97          0.0 pp           0.0%
                             Value Seekers      $108       $113          7.0 pp          26.9%
                    -------------------------------------------------------------------------------------
                    ```
                """, className="text-content", style={'margin-top': '0'}),

            # --- END INSERTED TABLE ---

            html.P([html.Strong("", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
            dcc.Markdown("""
                    **Outcome:** *Feature Explanation Generally Boosts WTP*

                    For most segments, being able to receive an “explanation” of the hands-free tech resulted in a higher price point at which 50% or more of respondents were willing to pay.

                    Of course, though, it varied by segment as you can see in the chart:
                    * Premium Enthusiasts showed the largest increase in WTP - a full $25 lift. They were also the group that had the highest baseline.
                    * Interested Pragmatists at $16, Curious Skeptics at $15, and Value Seekers at $5 (Traditionalists at $0).

                    This tracks with what we know regarding consumer behavior: we’re likely to pay more when a product we’re considering offers significantly more convenience or performance than others in the same category. Not only that, but we tend to justify those premiums when the innovation that leads to that enhanced convenience is well-communicated and clearly explained.
                """, className="text-content", style={'margin-top': '0'}),

            # --- END COPY THIS BELOW FOR STRONG SECTIONS
                html.P([html.Strong("Is out-of-state WTP so low because of low out-of-state brand awareness?", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '2rem 0 0.5rem 0'}),
                dcc.Markdown("""

                    **Outcome:** *Brand Awareness *does* increase WTP, but relatively less than other factors*

                    Many regions showed meaningful differences in WTP post-explanation. Overall, the West and Northeast demonstrated higher baseline WTP percent

                    Our multi-linear regression model (the “Key Statistical Drivers of WTP” chart) demonstrated that awareness truly does matter, though not as much as other factors.

                    In the regression, brand awareness is labeled as BrandAwarenessProxyDistWT.

                    **Overall impact of brand awareness on WTP:**
                    * Positive correlation: from the graph, we can see that there’s a positive correlation, meaning that higher brand awareness does predict a higher WTP
                    * Statistical Significance: the p-value is less than 0.05 (at 0.038), which means that we can be reasonably confident that this correlation is not due to just random chance alone ( at least based on the way it’s been modeled here).
                    * Strength/Magnitude of the effect: at 4.53, its coefficient is much less than that of other predictor variables, namely being in the West, Northeast, etc. So there is a relationship, but it’s just not as *strong* as a relationship with other factors.

                    So overall, the actual impact of this awareness proxy appears to be fairly modest in this specific study. Again, having additional ad data and other valuable marketing metrics to really inform brand awareness would most likely bolster the effect it has on WTP.
                   
                    """, className="text-content", style={'margin-top': '2rem !important'}),

                html.P([html.Strong("West and Northeast Provide Higher WTP Baseline Percentages", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '2rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    
                    Even after accounting for demographics and awareness, the West and Northeast show significant positive coefficients, moreso than the South or Midwest, again meaning that someone from the West or Northeast is more likely to have a higher willingness to pay after having the feature “explained” to them than other areas, even while taking into account all the other predictor factors like income bracket, age, etc. 
                    (Analysis based on $139 target price; Kizik’s median women's non-sale shoe price is ~$132).

                    FYI: We’re using the US Census’ regional mapping, with the West being 13 states: 

                    *Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, and Wyoming.*

                    And for the Northeast, the following nine states: 

                    *Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont, New Jersey, New York, and Pennsylvania.*

                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Different Segments Require Different Approaches", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '2rem 0 0.5rem 0'}),
                dcc.Markdown("""    
                    
                    The varying segments display different purchase motivations and ideal usage scenarios.

                    The Top 3 Drivers chart shows the differences in priorities across segments. For example, "Value Seekers" heavily prioritize Price and Durability, while "Premium Enthusiasts" place much higher emphasis on HandsFree Convenience and potentially Style or Brand. "Traditionalists" show very low interest in the core tech feature.

                    The Primary Usage chart provides valuable insight into their behaviors. “Everyday casual wear” is popular across all segments, while “Specific activity needing convenience” is much higher for Curious Skeptics and Interested Pragmatists.

                    Kizik doesn’t do this with their marketing, but it’s pretty clear that one-size-fits-all marketing would definitely not work given the significant variation in usage. It should definitely be finely tailored toward each segment, with a focus on innovation and style for Premium Enthusiasts, and an emphasis on convenience/ease-of-use for our Curious Skeptics/Interested Pragmatists.
                    
                    """, className="text-content", style={'margin-top': '0'}),


            ], className="text-container", style={'margin-bottom': '3rem'}), # Standard bottom margin after text block
        ], id="key-findings", className="text-container"),

        html.Div([
            html.H3("Business Impact & Strategic Recommendations", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""
                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""    
                    
                    Overall, the data provides rich context into different areas for expansion, and opportunities to serve current segments with products that better align with their preferences, simulated WTP, and budget.
                    
                    In this section, we'll start at the regional level and then narrow in on the state level.
                
                    """, className="text-content", style={'margin-top': '0'}),
    
                # --- Expansion Priority Matrix ---
            html.Div(dcc.Graph(id='expansion-matrix-chart', config=STANDARD_GRAPH_CONFIG), 
                     className="chart-card", 
                     style={'margin-top': '4rem'},
                     id='expansion-matrix-container'),
            # MODIFIED TOOLBAR STRUCTURE (WITH BOTH CONTROLS)
            html.Div(className="static-chart-toolbar", children=[
                html.Div(className="toolbar-controls-wrapper", children=[
                    # Segment Group
                    html.Div(className="toolbar-control-group segment-group", children=[
                        dbc.Label("Customer Segment", className="toolbar-label"),
                        html.Div(id={'type': 'static-segment-options', 'chart': 'expansion-matrix'}, className='toolbar-options-list')
                    ]),
                    # Divider
                    html.Div(className="toolbar-divider"),
                    # WTP Group
                    html.Div(className="toolbar-control-group wtp-group", children=[
                        dbc.Label("WTP Price Point", className="toolbar-label"),
                        html.Div(id={'type': 'static-wtp-options', 'chart': 'expansion-matrix'}, className='toolbar-options-grid')
                    ])
                ])
            ]),

            html.Div([ 
                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    The Expansion Priority Matrix identifies key regions for market growth by plotting Market Attractiveness against Market Entry Opportunity. To make things more concrete, if you wanted to find a market for expansion that would respond well to Kizik products, you'd want to find something in the top right quadrant. There, you'll find the perfect blend of people willing to pay higher prices who also have less brand awareness, or more room to grow brand recognition.

                    **Quadrants**

                    Overall, the quadrants in this matrix can help a researcher understand which areas to avoid and which areas are worth potential future investment.

                    *"High Priority"* (Top right: high attractiveness, high opportunity)

                    *"Optimize"* (Bottom right: high attractiveness, low opportunity)

                    *"Monitor"* (Top left: low attractiveness, high opportunity)

                    *"Lower Priority"* (Bottom left: attractiveness, low opportunity)

                    * **Market Attractiveness (X-axis)** shows the percentage of respondents in a region willing to pay the selected price point, indicating immediate revenue potential.

                    * **Market Entry Opportunity (Y-axis)** represents the inverse of current brand awareness (using the refined Brand Awareness proxy metric), highlighting areas with the most room to grow brand recognition. Higher scores mean lower current awareness, which again means that there's significant room to grow and improve brand awareness.
                """, className="text-content", style={'margin-top': '0'}),
                
                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""
                    At $139 and $159 (both common price points for Kizik's current offerings), the West and Northeast regions are the most attractive. The Northeast is fairly unique, in that it has both a higher brand awareness and a high willingness to pay, moreso than the West. It also has three flagship stores, while the West only has two. The West as well represents a great opportunity for continuous expansion, not to mention relatively reduced shipping times from their warehouse in Lindon.
                """, className="text-content", style={'margin-top': '0'})
            ], className="text-container"),


                html.P([html.Strong("State-Specific Expansion Opportunities", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                    dcc.Markdown("""""", className="text-content", style={'margin-top': '0'}),

                html.P("Top 5 States For Potential Expansion (Excluding UT)", style={'text-align': 'center', 'font-weight': 'bold', 'margin-top': '1rem', 'margin-bottom': '0.5rem'}),
                    dcc.Markdown("""
                        ```text
                        | State | Avg       | Opportunity     | WTP %  | Respondent | Rank_Opp | Rank_WTP | Rank_Size | Combined|
                        |       | Awareness | (1-AvgAwareness)| @ $139 | Count      |          |          |           | Rank    |
                        |-------|-----------|-----------------|--------|------------|----------|----------|-----------|---------|
                        | CA    | 0.15      | 0.85            | 65%    | 50         | 3        | 1        | 1         | 5       |
                        | NY    | 0.12      | 0.88            | 60%    | 45         | 2        | 2        | 2         | 6       |
                        | TX    | 0.10      | 0.90            | 55%    | 48         | 1        | 4        | 1         | 6       |
                        | FL    | 0.18      | 0.82            | 58%    | 40         | 4        | 3        | 3         | 10      |
                        | IL    | 0.08      | 0.92            | 50%    | 30         | 0        | 6        | 5         | 11      |
                        ```
                
                        """, className="text-content", style={'margin-top': '0',}),

                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""    
                    
                        This chart represents the top 5 states (outside of Utah) for investment based on *potential* revenue generation - based on the math behind the Expansion Matrix chart (again, that chart just shows Market Attractiveness + Market Opportunity).

                        In this scenario, the lower the score, the higher the potential for investment.

                        **Before we get into the nitty-gritty of what each column represents, here's what we're looking for:**

                        * **High Market Attractiveness:** states with a high percentage of the population that already has a high WTP ($149 to $159) after understanding the feature

                        * **High Market Opportunity:** states with a low brand awareness, meaning high growth potential.

                        * **Fairly High Population (Market Size): states with higher populations that aren't necessarily the "highest" in market attractiveness or opportunity will still outperform smaller states that are higher in those areas.

                        **Additional details:**
                        * $139 is the WTP price point used here - again, Kizik’s median women's non-sale shoe price is ~$132.
                        * Test Group is used here as well

                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                dcc.Markdown("""    
                    
                        **Ranking:**
                        * CA: Highest WTP, pretty decent opportunity, largest size. Strong candidate.
                        * TX: Highest opportunity, strong size, decent WTP. Strong candidate.
                        * NY: Decent all-arounder, possibly less opportunity/WTP than TX/CA but still fairly strong.
                        * FL: Slightly lower opportunity and WTP than the top 3, but still decent size.   

                       There's decent opportunity across these states to expand, taking advantage of high WTP, low awareness, and fairly large population size. Kizik is still a fairly new brand, so these states represent a potential great first place to start when evaluating expansion areas (again, according to our simulation).        
                    """, className="text-content", style={'margin-top': '0'}),

                
                html.P( # This is the P component for the title
                                    [ # Children are now a list
                                        "Segment-Product Portfolio Alignment ", # First part of the title
                                        html.Br(),                             # Line break
                                        "(Based on Test Group PME & Simulated Even Segment Distribution)" # Second part
                                    ], 
                                    style={'text-align': 'center', 'font-weight': 'bold', 'margin-top': '3rem', 'margin-bottom': '0.5rem'}
                                ),                    dcc.Markdown("""

                        ```text
                        -----------------------------------------------------------------------------------------------------------
                        Segment                | % of Sim. | Market WTP Ceiling | # Models      | # Outdoor | # Active | # Casual |
                                               |           | (PME - Test)       | <= WTP Ceiling| <= WTP    | <= WTP   | <= WTP   |
                        -----------------------|-----------|--------------------|---------------|-----------|----------|----------|
                        Curious Skeptics       | 20.0%     | $130               | 11            | 0         | 3        | 8        |
                        -----------------------|-----------|--------------------|---------------|-----------|----------|----------|
                        Interested Pragmatists | 20.0%     | $128               | 10            | 0         | 2        | 8        |
                        -----------------------|-----------|--------------------|---------------|-----------|----------|----------|
                        Premium Enthusiasts    | 20.0%     | $140               | 13            | 0         | 5        | 8        |
                        -----------------------|-----------|--------------------|---------------|-----------|----------|----------|
                        Traditionalists        | 20.0%     | $137               | 12            | 0         | 4        | 8        |
                        -----------------------|-----------|--------------------|---------------|-----------|----------|----------|
                        Value Seekers          | 20.0%     | $134               | 11            | 0         | 3        | 8        |
                        -----------------------------------------------------------------------------------------------------------

                        """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Overview", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                    dcc.Markdown("""   

                    Kizik offers a wide variety of shoe models, but, like any line of products, not all of those models will fit certain budgets or preferences. This analysis focuses on analyzing the relationship between the number of models available to each segment (utilizing the Test group only) at or equal to their highest WTP threshold. 

                    **Currently, Kizik groups their models by 3 core categories: Active, Relaxed, and Explore.**

                    I generated this chart in order to understand which categories might be under or over-represented for each Segment, allowing me to understand gaps in the current product portfolio that could be filled with shoe models better aligned with current segment preferences and WTP.

                    **Additional details:**
                    * This analysis focused on the women’s line only, 22 models across 3 categories
                    * Currently, the women’s line has a median price of $132: the WTP point I chose to generate data for the chart was $139 as it was the closest price point
                    * Given this is a parameter-informed simulation, the current max global ceiling for WTP is $140 from the Premium Enthusiasts, and the current max price on any model offered from Kizik’s site are the Junos at $169. Why the discrepancy? As a simulation with the parameters I’ve set, it’s not going to be able to take in the qualitative, experiential inputs that go into true value perception (seeing try-on videos from social media, word-of-mouth from friends, additional forms of paid advertising, etc.

                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Analysis", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                    dcc.Markdown("""
                    There are decent opportunities both inside and outside the price bands that Kizik has set. At the moment, the majority of the higher prices contain more "Explore" models and less "Active" models, and there's ample opportunity to expand offerings in those categories to segments at lower price points.
                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Clear Lack of Entry to Mid-Price Explore Models", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                    dcc.Markdown("""   

                    Even without my chart, it’s fairly clear from a quick look at Kizik’s website that there aren’t many Explore options below $149 - in fact, there are none. 

                    While it looks from this disparity like Kizik has defined price itself as a category, it would be reasonable to assume that there’s a decent opportunity to introduce some kind of an “Explore Light” or “Active Trail” model to capture demand that could exist at or below that level.
                    
                    Priced around the $125 - $140 price range, Kizik could riff off of an already popular Active model, the Athens  (priced at $129 on the site), and call it the Athens Trail, positioning it for Premium Enthusiasts (and most likely capturing demand from other overlapping segments).
                    
                    By pricing it lower and associating it with one of their most popular models, there’s a chance they could safely boost revenue at that price range without sacrificing model familiarity. It could serve as a lower-priced alternative to the Wasatch, a rugged, trail-ready shoe currently priced at $169.

                    
                    """, className="text-content", style={'margin-top': '0'}),

                html.P([html.Strong("Premium Enthusiasts (PME $140): Consider Explore options, double down on Active features", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                    dcc.Markdown("""   

                    They have 10 models within their acceptable price range, with 8 Relax models and only 2 Active Models (0 Explore).

                    Quite clear that they’re fairly underserved in this regard - expanding the “Active” offerings in the $110 - $128 price range with features that are an appealing hybrid between casual/leisure and light athletic activity could fill this gap.                 
                    
                    """, className="text-content", style={'margin-top': '0'}),                    
                    
                html.P([html.Strong("Value Seekers (PME $134) and Curious Skeptics (PME $130)", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                    dcc.Markdown("""   

                    This group has 11 models available to them: 0 Outdoor, 3 Active, 8 Casual/Relaxed.

                    Currently priced out of Explore (or if they wanted more rugged, durable features) and limited with their Active options, there’s opportunity to create a lighter, entry level “Active Basic” or even a light-on-features “Outdoor Value” model priced fairly aggressively at the $119 - $129 range.
                    
                    This could do fairly well in newer markets where price tolerance and overall brand trust aren’t very high.                    
                    
                    """, className="text-content", style={'margin-top': '0'}),     

                html.P([html.Strong("Summary", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '3rem 0 0.5rem 0'}),
                    dcc.Markdown("""   

                From this data, we can see that there's possibly untapped potential within Kizik's current offerings. While price definitely seems to define its own categories (Explore), it stands to reason that there might be more affordable versions of their more successful models
                    """, className="text-content", style={'margin-top': '0'}),             
            
        ], id="business-impact", className="text-container"),


    
        html.Div([
            html.H3("Limitations", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            # Keep the original Markdown as an intro/summary if desired, or remove it
            dcc.Markdown("""
                    This section details the key limitations of this project, primarily stemming from its reliance on simulated data and specific proxy metrics rather than direct field research or access to internal company data. Understanding these limitations is crucial for interpreting the findings appropriately.
                    
                    Below are the limitations in depth, but here is an additional summary in the interest of time:
                    * **Nature of the study:** It's a simulation, and can't produce true synthetic data.
                    * **Conceptual explanation vs physical:** Real WTP uplift would be captured in an in-person trial
                    * **Awareness proxy:** Real-world research would use direct questions/social media to determine awareness
                    * **Competitive landscape:** Presence of real competitors would more accurately adjust WTP
                    * **Segmentation Logic:** Segments were pre-defined vs naturally emerging from the data
                
                """, className="text-content") # Changed original content to an intro paragraph
        ], id="limitations", className="text-container"),
        # --- NEW: Interactive Limitations Carousel ---
        html.Div([
            # NEW: Title Display Area (populated by callback)
            html.H4(
                limitations_content[0]['title'], # Load default title
                id='limitation-title-display',
                className="limitation-info-title", # Use new class for consistency if needed
                style={'font-weight': 'bold', 'margin-bottom': '1rem', 'padding-left': '1rem'} # Basic title styling
            ),
            # Content Display Area (initially load the first limitation)
            html.Div(
                id='limitation-content-display',
                children=create_limitation_content(limitations_content[0]['id'], limitations_content) # Load first item by default
            ),
            # Limitation Selector Tabs
            html.Div(
                id='limitation-selector-tabs',
                className='limitation-selector-tabs',
                children=[
                    # Create a button for each limitation
                    html.Button(
                        item['title'], # Button text is the title
                        id={'type': 'limitation-tab', 'index': item['id']}, # Pattern-matching ID
                        className=f"limitation-tab-button {'active' if i == 0 else ''}", # Set first button active
                        n_clicks=0
                    ) for i, item in enumerate(limitations_content) # Iterate through the content list
                ]
            )
        ], id="limitations-info-container", className="limitations-info-container"), # Use new ID and class
        # --- END: Interactive Limitations Carousel ---

                # Data Section (Explicit Div for Button)
        html.Div([
            html.H3("Data and Sources", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600'}),

            # Wrap the button in its own Div
            html.Div([
                dbc.Button("Show Sample Data", id="collapse-button", className="mb-3", color="primary", n_clicks=0),
            ]), # End of Div wrapping the button

            dbc.Collapse(
                dbc.Card(dbc.CardBody(html.Div(id='sample-data-table'))),
                id="collapse",
                is_open=False
            ),
        ], id="data", className="text-container"),

        html.Div([ # Standard text container for the block
                html.P([html.Strong("Overview of Data Sources", style={'font-size': '18px', 'font-weight': 'bold'})], style={'margin': '1.5rem 0 0.5rem 0'}),
                dcc.Markdown("""

                    **Underlying Data Sources for Synthetic Data Parameters**

                    It's important to remember that the data used in the creation of this model contains simulated data. However, the parameters used to *generate* this data were informed by publicly available sources to make the simulation realistic.

                    **U.S. Census Bureau (Demographics & Regional Distribution)**
                    * **Link:** https://www.census.gov/data.html (Data portal); Specific tables often accessed via American Community Survey (ACS) data explorers.
                    * **Why Used:** To ensure the synthetic respondent pool realistically reflects the US population in terms of age, gender, income distribution, and geographic spread (regional population weights). Makes the sample nationally representative.
                    * **Specific Data/Variables Used to Inform Parameters:**
                    * Population estimates by Age Group (used for age_weights).
                    * Population estimates by Gender (used for gender_sample weights).
                    * Household Income distribution data (used for income_weights).
                    * Population estimates by Region/State (used for region_pop_weights and ensuring state distribution roughly matched regional population sizes).

                    **2. Bureau of Labor Statistics (BLS) Consumer Expenditure Survey (CEX)**
                    * **Link:** https://www.bls.gov/cex/
                    * **Why Used:** To ground assumptions about general footwear spending patterns and price points in reality. Provides benchmarks for how much consumers typically spend on apparel/footwear across different demographics.
                    * **Specific Data/Variables Used to Inform Parameters:**
                    * Average expenditure on Footwear by income bracket and age group (informed the simulation of RecentPurchasePrice and the general range for base_wtp_sample).
                    * General price distributions for casual footwear categories (helped set plausible ranges for VW and Gabor-Granger price points).

                    **3. Kizik Store Locator Data (Physical Distribution)**
                    * **Link:** Kizik Store Locator
                    * **Why Used:** Provided the ground truth for Kizik's actual physical retail presence (the 372 store locations). This was essential for calculating the store density and the modified distance-weighted brand awareness proxy logic (taking into account flagship store location vs retail, adjacency bonus, distance decay, etc).
                    * **Specific Data/Variables Used:**
                    * List of store locations (City, State) used to count stores per state (store_counts).
                    * This data, combined with population data, was used to calculate stores_per_100k and ultimately the BrandAwarenessProxy_DistWt.
                    * Informed the simulation of Q10_2_DistNearestStoreMiles.

                    **Summary**

                    The survey questions aim to comprehensively measure price sensitivity, understand its drivers, and assess the impact of feature education within a realistic market context. The synthetic data itself is artificial, but its characteristics (demographics, price ranges, awareness levels) were carefully parameterized based on real-world patterns observed in Census, BLS CEX, and Kizik's actual store distribution data to ensure the simulation yields plausible and defensible insights.
                    * *Technical Summary by Gemini*
                """, className="text-content", style={'margin-top': '0'}),
            ], className="text-container", style={'margin-bottom': '3rem'}), # Standard bottom margin after text block


        html.Div([
            html.H3("Technology Stack", className="mb-2", style={'font-size': '1.5rem', 'font-weight': '600', 'width': '100%'}),
            dcc.Markdown("""
            
            I used Python, specifically Dash to build this site and Plotly to create the charts. I want to be *very* clear that I don't have a background in computer science; this was just a lot of trial and error and back and forth with the AI models to develop the code and make sure it worked and displayed data accurately.
            
            AI models used were Claude Pro by Anthropic, Gemini 2.5 Pro Experimental 03-25 and Preview 03-25 for more intense data refinement and code troubleshooting, as well as Grok 3/SuperGrok for some minor tasks. I additionally used Cursor Pro and Windsurf Pro as my IDEs and a lot of patience to build this site :)""", className="text-content")
        ], id="tech-stack", className="text-container"),
        # --- END NEW SECTIONS ---

    ], id="page-content",
    style=CONTENT_STYLE
    ) 
]) 
print("App layout defined.")

# Define the callback function
# Output list and function signature remain the same as the previous version
@app.callback(
    [
        Output("van-westendorp-chart-control", "figure"),
        Output("van-westendorp-chart-test", "figure"),
        Output("wtp-gg-chart", "figure"),
        Output("regional-wtp-map", "figure"),
        Output("expansion-matrix-chart", "figure"),
        Output("regression-coef-plot", "figure"),
        Output("top-drivers-chart", "figure"),
        Output("primary-usage-chart", "figure")
    ],
    [
        Input("selected-values-store", "data")
    ]
)
def update_visualizations(selected_values): 
    triggered_id = ctx.triggered_id
    print(f"Update visualizations triggered by: {triggered_id}")

    no_data_fig = go.Figure().update_layout(title_text="Data not available", xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='white', paper_bgcolor='white')
    num_outputs = 8 

    if app_data is None: 
        print("Callback update_visualizations skipped: Global data not available.")
        return [no_data_fig] * num_outputs

    if selected_values is None:
        print("Update Viz: No selected values.")
        return [no_data_fig] * num_outputs

    error_fig = go.Figure().update_layout(title_text="Error generating chart", xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='white', paper_bgcolor='white')
    vw_fig_control, vw_fig_test, wtp_fig, map_fig, matrix_fig, reg_fig, drivers_fig, usage_fig = [error_fig] * num_outputs

    try:
        selected_segment = selected_values.get('segment', 'All Segments')
        selected_price = selected_values.get('wtp', 139)
        df = app_data 
        if df.empty:
            print("Update Viz: Global dataframe is empty.") 
            return [no_data_fig] * num_outputs

        print(f"Update Viz: Rendering cross-segment charts")
        reg_fig = create_regression_coef_plot() 
        drivers_fig = create_top_drivers_chart() 
        usage_fig = create_primary_usage_chart() 

        print(f"Update Viz: Rendering filterable charts for Segment='{selected_segment}', Price=${selected_price}")
        vw_fig_control = create_vw_chart(selected_segment, group_filter='Control') 
        vw_fig_test = create_vw_chart(selected_segment, group_filter='Test') 
        wtp_fig = create_wtp_gg_chart(selected_segment) 
        map_fig = create_regional_map(selected_segment, selected_price) 
        matrix_fig = create_expansion_matrix(selected_segment, selected_price) 

    except Exception as e:
        print(f"Error during visualization update: {e}")
        import traceback
        traceback.print_exc()
        return [error_fig] * num_outputs

    return vw_fig_control, vw_fig_test, wtp_fig, map_fig, matrix_fig, reg_fig, drivers_fig, usage_fig


# Callbacks for sidebar collapse (Keep)
@app.callback(Output("collapse-overview", "is_open"), Input("nav-overview", "n_clicks"), State("collapse-overview", "is_open"), prevent_initial_call=True)
def toggle_overview_collapse(n, is_open): return not is_open if n else is_open
@app.callback(Output("collapse-methodology", "is_open"), Input("nav-methodology-design", "n_clicks"), State("collapse-methodology", "is_open"), prevent_initial_call=True)
def toggle_methodology_collapse(n, is_open): return not is_open if n else is_open
@app.callback(Output("collapse-my-role", "is_open"), Input("nav-my-role", "n_clicks"), State("collapse-my-role", "is_open"), prevent_initial_call=True)
def toggle_my_role_collapse(n, is_open): return not is_open if n else is_open

# NEW Callback for Visualizations collapse
@app.callback(Output("collapse-visualizations", "is_open"), Input("nav-visualizations", "n_clicks"), State("collapse-visualizations", "is_open"), prevent_initial_call=True)
def toggle_visualizations_collapse(n, is_open): return not is_open if n else is_open

# Callback for Sample Data Table (Keep)
@app.callback(
    Output("collapse", "is_open"),
    Output("sample-data-table", "children"),
    Input("collapse-button", "n_clicks"),
    State("collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_collapse_and_show_data(n, is_open): 
    if n:
        table_content = "Global data could not be loaded." 
        if app_data is not None: 
            try:
                sample_df = app_data.head(10)
                table_content = html.Div(dbc.Table.from_dataframe(sample_df, striped=True, bordered=True, hover=True, responsive=False, className='table-sm'), style={'overflowX': 'auto'})
            except Exception as e:
                print(f"Error creating sample data table from global app_data: {e}")
                table_content = f"Error displaying data: {e}"
        return not is_open, table_content
    return is_open, dash.no_update

# Callback for Carousel Caption (Keep)
@app.callback(Output("external-carousel-caption", "children"), Input("awareness-proxy-carousel", "active_index"))
def update_external_caption(active_index):
    carousel_captions = [
        "Trying to find a good proxy for brand awareness",
        "Evaluating proposed proxies",
        "Determining defensibility of store proximity as a proxy",
        "Exploring ways to strengthen store proximity",
        "Adding distance weighting",
    ]
    if active_index is None: active_index = 0
    return carousel_captions[active_index] if 0 <= active_index < len(carousel_captions) else ""

# --- Clientside Callbacks (Active Links, Scroll - Keep As Is) ---
# --- Clientside callback for active links ---
# Clientside callbacks remain the same as the previous version, handling navigation and active states
app.clientside_callback(
    """
    function(pathname, hash) {
        // Function to determine the currently visible section
        function getCurrentSection() {
            // MODIFIED: Update section querySelector to include new VW chart containers AND LIMITATIONS ID
            const sections = document.querySelectorAll('#overview, #goal, #objective, #observations, #approach, #methodology-design, #gabor-granger, #wtp, #van-westendorp, #control-group, #competitive-adjustment, #population-sampling, #awareness-proxy, #survey-flow, #my-role, #rejecting-methodology, #identifying-ambiguity, #refining-proxy, #pushing-for-deeper-portfolio-analysis, #visualizations, #segment-profiles, #key-findings, #business-impact, #limitations, #limitations-info-container, #data, #tech-stack, #vw-chart-control-container, #vw-chart-test-container, #wtp-gg-chart-container, #regional-map-container, #expansion-matrix-container, #regression-coef-container, #top-drivers-container, #primary-usage-container, #segment-info-container');
            let currentSectionId = null;
            let minDistance = Infinity;
            const activationThreshold = 150; // How close to top before activating
            const scrollContainer = document.getElementById('page-content');
            const scrollY = scrollContainer ? scrollContainer.scrollTop : window.pageYOffset;
            const containerRect = scrollContainer ? scrollContainer.getBoundingClientRect() : null;

            if (hash && hash !== '#') {
                const potentialId = hash.substring(1);
                const targetElement = document.getElementById(potentialId);
                 if (targetElement) {
                     let isValidSection = false;
                     // Check if the hashed ID is one of our main scroll target sections
                     // This includes section containers AND specific elements like H3s if they are direct href targets
                     sections.forEach(sec => { if (sec.id === potentialId) isValidSection = true; });
                     
                     // Also consider if the hash directly matches an ID even if not in `sections` querySelector explicitly,
                     // as long as it's a valid target on the page.
                     if (!isValidSection && document.getElementById(potentialId)) {
                        // If it's a valid element ID on the page, we can use it.
                        // The scroll-based logic later will refine if another "section" is more prominent.
                        isValidSection = true; 
                     }

                     if (isValidSection) {
                         currentSectionId = potentialId;
                         // If a specific chart or interactive container is hashed, prioritize it
                         if (potentialId.includes('-container')) { 
                              return currentSectionId;
                         }
                     }
                 }
             }

            // If no hash or hash wasn't a valid section, determine by scroll position
            if (!currentSectionId && scrollContainer && containerRect) {
                if (scrollY < 50 && sections.length > 0) {
                    currentSectionId = sections[0].id; // Default to first section if at top
                } else {
                     // Prioritize chart/interactive containers if they are near the vertical center
                     const viewportCenterY = containerRect.top + containerRect.height / 2;
                     let centerMostInteractiveId = null;
                     let minCenterDistance = Infinity;

                    sections.forEach(section => {
                        if (!section || !section.id) return;
                        const rect = section.getBoundingClientRect();
                        const topRelativeToContainer = rect.top - containerRect.top;
                        const sectionCenterY = rect.top + rect.height / 2;
                        const distanceToCenter = Math.abs(sectionCenterY - viewportCenterY);

                        // Check if section top is within activation threshold OR if it's an interactive element near center
                         if (topRelativeToContainer <= activationThreshold && rect.bottom > activationThreshold / 2) {
                              const distanceToTop = Math.abs(topRelativeToContainer);
                              if (distanceToTop < minDistance) {
                                 minDistance = distanceToTop;
                                 currentSectionId = section.id;
                             }
                         }

                         // Check for interactive element centering (chart containers, segment info, limitations info)
                         if (section.id.includes('-container')) { 
                            if(distanceToCenter < containerRect.height / 2) { // Check if element is within viewport vertically
                                if(distanceToCenter < minCenterDistance) {
                                    minCenterDistance = distanceToCenter;
                                    centerMostInteractiveId = section.id;
                                }
                             }
                         }
                    });

                    // Prefer the interactive element closest to the center if one was found and is reasonably close
                    if (centerMostInteractiveId && minCenterDistance < 150) { // Add threshold for center proximity
                        currentSectionId = centerMostInteractiveId;
                    }
                    // Fallback if scrolled down but no section met criteria by top position
                     if (!currentSectionId && scrollY > activationThreshold) {
                         for (let i = sections.length - 1; i >= 0; i--) {
                            if(sections[i]){ // Check element exists
                                const rect = sections[i].getBoundingClientRect();
                                const topRelativeToContainer = rect.top - containerRect.top;
                                if (topRelativeToContainer < activationThreshold) {
                                    currentSectionId = sections[i].id;
                                    break;
                                }
                            }
                         }
                     }
                }
            }

            // Handle reaching bottom of page
            if (!currentSectionId && scrollContainer && scrollY > 100) {
                const bottomOfPage = (scrollContainer.scrollTop + scrollContainer.clientHeight) >= scrollContainer.scrollHeight - 50;
                if (bottomOfPage && sections.length > 0) {
                    let lastSection = sections[sections.length - 1]; // Default to last in querySelector
                    // Try to find the last *main content* section, not just a sub-component like a chart container, if possible.
                    // This depends on your definition of "main content section".
                    // For now, the last element in `sections` list that's visible is good enough.
                    currentSectionId = lastSection ? lastSection.id : sections[sections.length - 1].id;
                }
            }

            // Default to first section if still nothing found
            if (!currentSectionId && sections.length > 0) {
                currentSectionId = sections[0].id;
            }
            // console.log("Current Section:", currentSectionId);
            return currentSectionId;
        }

        const currentSectionId = getCurrentSection();
         const idMap = { // UPDATED idMap
            'executive-summary': 'nav-executive-summary',   
            'overview': 'nav-overview',
            'goal': 'nav-goal',
            'objective': 'nav-objective',
            'observations': 'nav-observations',
            'approach': 'nav-approach',

            'methodology-design': 'nav-methodology-design',
            'gabor-granger': 'nav-gabor-granger',
            'wtp': 'nav-wtp', // Added
            'van-westendorp': 'nav-van-westendorp',
            'control-group': 'nav-control-group',
            'competitive-adjustment': 'nav-competitive-adjustment',
            'population-sampling': 'nav-population-sampling',
            'awareness-proxy': 'nav-awareness-proxy',

            'my-role': 'nav-my-role',
            'rejecting-methodology': 'nav-rejecting-methodology',
            'identifying-ambiguity': 'nav-identifying-ambiguity',
            'refining-proxy': 'nav-refining-proxy',
            'pushing-for-deeper-portfolio-analysis': 'nav-pushing-for-deeper-portfolio-analysis',

            'survey-flow':'nav-survey-flow',

            'visualizations': 'nav-visualizations', 
            'vw-chart-control-container': 'nav-vw-chart',
            'vw-chart-test-container': 'nav-vw-chart', 
            'wtp-gg-chart-container': 'nav-wtp-gg-chart',
            'regional-map-container': 'nav-regional-map',
            'expansion-matrix-container': null, // No nav link for this
            'regression-coef-container': 'nav-regression-coef',
            'top-drivers-container': 'nav-top-drivers',
            'primary-usage-container': 'nav-primary-usage',

            'segment-profiles': 'nav-segment-profiles', // Main section title div
            'segment-info-container': 'nav-segment-profiles', // Interactive container

            'key-findings': 'nav-key-findings',
            'business-impact': 'nav-business-impact',

            'limitations': 'nav-limitations', // Main section title div
            'limitations-info-container': 'nav-limitations',

            'data': 'nav-data',
            'tech-stack': 'nav-tech-stack'
        };
         const navLinkOrder = [ // Must match Python Output list exactly
            'nav-overview', 'nav-executive-summary', 'nav-goal', 'nav-objective', 'nav-observations', 'nav-approach',
            'nav-methodology-design', 'nav-gabor-granger', 'nav-van-westendorp',
            'nav-control-group', 'nav-competitive-adjustment', 'nav-population-sampling',
            'nav-awareness-proxy', 'nav-survey-flow', 'nav-my-role',
            'nav-rejecting-methodology', 'nav-identifying-ambiguity',
            'nav-refining-proxy', 'nav-pushing-for-deeper-portfolio-analysis',
            'nav-visualizations',
            'nav-vw-chart', 'nav-wtp-gg-chart', 'nav-regional-map', 'nav-regression-coef', 'nav-top-drivers', 'nav-primary-usage',
            'nav-segment-profiles', // Added to match Python Output
            'nav-key-findings', 'nav-business-impact', 'nav-limitations', 'nav-data', 'nav-tech-stack'
        ];

           const activeNavLinkId = idMap[currentSectionId] || null;
           const activeStatuses = navLinkOrder.map(linkId => {
               if (!activeNavLinkId) return false; // No active section, so no link is active

               // Rule 1: If this linkId is the one directly corresponding to the active section, it's active.
               if (linkId === activeNavLinkId) return true;

               // Rule 2: Parent highlighting logic
               if (linkId === 'nav-overview' && ['nav-executive-summary', 'nav-goal', 'nav-objective', 'nav-observations', 'nav-approach'].includes(activeNavLinkId)) return true;
               if (linkId === 'nav-methodology-design' && ['nav-gabor-granger', 'nav-wtp', 'nav-van-westendorp', 'nav-control-group', 'nav-competitive-adjustment', 'nav-population-sampling', 'nav-awareness-proxy'].includes(activeNavLinkId)) return true;
               if (linkId === 'nav-my-role' && ['nav-rejecting-methodology', 'nav-identifying-ambiguity', 'nav-refining-proxy', 'nav-pushing-for-deeper-portfolio-analysis'].includes(activeNavLinkId)) return true;
               if (linkId === 'nav-visualizations' && ['nav-vw-chart', 'nav-wtp-gg-chart', 'nav-regional-map', 'nav-regression-coef', 'nav-top-drivers', 'nav-primary-usage'].includes(activeNavLinkId)) return true;
               
               // 'nav-segment-profiles', 'nav-key-findings', etc. are top-level and handled by Rule 1.
               // No other parent-child rules needed if the nav structure is flat beyond these.

               return false;
           });
           // console.log("Active Statuses:", activeStatuses);
           return activeStatuses;
        }
    """,
    [ # UPDATED Python Output list to match JS navLinkOrder (31 items)
        Output("nav-overview", "active"), Output("nav-executive-summary", "active"), Output("nav-goal", "active"), Output("nav-objective", "active"),
        Output("nav-observations", "active"), Output("nav-approach", "active"),
        Output("nav-methodology-design", "active"), Output("nav-gabor-granger", "active"), Output("nav-van-westendorp", "active"),
        Output("nav-control-group", "active"), Output("nav-competitive-adjustment", "active"), Output("nav-population-sampling", "active"),
        Output("nav-awareness-proxy", "active"), Output("nav-survey-flow", "active"), Output("nav-my-role", "active"),
        Output("nav-rejecting-methodology", "active"), Output("nav-identifying-ambiguity", "active"), Output("nav-refining-proxy", "active"),
        Output("nav-pushing-for-deeper-portfolio-analysis", "active"),
        Output("nav-visualizations", "active"),
        Output("nav-vw-chart", "active"), Output("nav-wtp-gg-chart", "active"), Output("nav-regional-map", "active"),
        Output("nav-regression-coef", "active"), Output("nav-top-drivers", "active"), Output("nav-primary-usage", "active"),
        Output("nav-segment-profiles", "active"), # Added
        Output("nav-key-findings", "active"), Output("nav-business-impact", "active"), Output("nav-limitations", "active"),
        Output("nav-data", "active"), # Order matches JS
        Output("nav-tech-stack", "active"), # Order matches JS
    ],
    Input("url", "pathname"),
    Input("url", "hash"),
)


# Clientside for auto-expanding sections (Keep as is or refine if needed)
app.clientside_callback(
    """
    function(pathname, hash) {
        // No direct output needed, this just triggers observers etc.
        return '';
    }
    """,
    Output("dummy-output", "children"),
    Input("url", "pathname"),
    Input("url", "hash")
)


# --- Custom CSS (index_string - MODIFIED Toolbar CSS) ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Kizik Price Sensitivity Dashboard</title>
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            /* --- Base Styles (Keep Essential) --- */
            /* BRUTAL RESET - VERY FIRST THING */
            * {
                scroll-behavior: auto !important;
            }
            html {
                scroll-behavior: auto !important; /* CHANGED: No smooth scroll globally */
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            @media (prefers-reduced-motion: no-preference) {
                :root {
                    scroll-behavior: auto !important;
                }
                /* You might also need it for html directly if :root isn't enough */
                html {
                    scroll-behavior: auto !important;
                }
            }
            body {
                scroll-behavior: auto !important;
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #2B2B2B;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                background-color: #F7F9FC;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            #page-content {
                margin-left: 18rem !important;
                margin-right: 0 !important;
                padding: 2rem 1rem 2rem 1rem !important;
                background-color: white !important;
                position: fixed !important;
                top: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                left: 0 !important;
                overflow-y: scroll !important;
                width: calc(100% - 18rem) !important;
                scroll-behavior: auto !important; /* CHANGED: Ensure container scroll is instant */
            }
            .dashboard-sidebar {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                bottom: 0 !important;
                width: 18rem !important;
                padding: 1.5rem 1rem !important;
                background-color: white !important;
                overflow-y: auto !important;
                z-index: 1000 !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
                outline: none !important;
                border-right: none !important;
                direction: ltr !important;
            }
            
            /* --- Segment Info Container Styles --- */
            .segment-info-container { /* Changed ID to class */
                padding: 0 1rem 0 1rem !important; /* Reduce padding */
                border: 1px solid #DEE2E6 !important;
                border-radius: 8px !important;
                background-color: white !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
                position: relative !important;
                overflow: hidden !important; /* Prevent container overflow */
                /* Sizing and Layout */
                min-height: 650px !important; /* Set min height like segments */
                height: 650px !important; /* Use fixed height */
                display: flex !important;
                flex-direction: column !important;
                max-width: 900px !important; /* Match chart card width */
                margin-left: auto !important; /* Center */
                margin-right: auto !important; /* Center */
                margin-top: 0.1rem !important;
                margin-bottom: 2rem !important;
            }

            #segment-title-display { /* Styling for the dynamic title */
               font-weight: bold !important;
               margin-bottom: 1rem !important;
               padding-left: 0 !important; /* Remove specific padding if container has padding */
               padding-top: 1.5rem !important; /* Add padding top to space it */
               flex-shrink: 0; /* Prevent title from shrinking */
            }

            #segment-content-display {
                height: calc(100% - 60px - 50px) !important; /* Height minus tabs and approximate title height */
                overflow-y: auto !important;
                padding-right: 0.5rem !important;
                margin-bottom: 0 !important;
                flex-grow: 1; /* Allow content to take remaining space */
                padding-left: 0 !important;
            }

            .segment-selector-tabs {
                position: static !important; /* No longer absolute */
                bottom: auto !important;
                left: auto !important;
                right: auto !important;
                padding: 0.75rem 0rem !important; /* Adjust padding */
                background: white !important;
                border-top: 1px solid #DEE2E6 !important;
                margin-top: auto !important; /* Push to bottom */
                display: flex !important;
                justify-content: space-around !important;
                flex-shrink: 0; /* Prevent tabs from shrinking */
            }
            
            /* --- Other App Styles (Keep Essential) --- */
            .text-container { max-width: 780px !important; width: 100% !important; margin-bottom: 2rem !important; margin-left: auto; margin-right: auto;}
            .text-content { max-width: 780px !important; width: 100% !important; margin-top: 12px !important; margin-right: 0 !important; margin-left: 0 !important; white-space: normal !important; word-break: break-word !important; display: block !important; line-height: 1.65 !important; font-weight: 400 !important; color: #3E3E3E !important; font-size: 15px !important; }
            .chart-card { background: white !important; border-radius: 12px !important; padding: 1.75rem 1.75rem 1.75rem 1.75rem !important; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important; width: 100% !important; max-width: 900px !important; border: 1px solid rgba(0, 0, 0, 0.03) !important; position: relative !important; /* margin-bottom: 0 !important; /* Removed default bottom margin */ margin-left: auto !important; margin-right: auto !important; } /* Centered BY DEFAULT */
            #carousel-card { padding-bottom: 75px !important; margin-bottom: 0 !important; position: relative !important; max-width: 85% !important; margin-right: auto !important; margin-left: auto !important; width: 85% !important; padding: 1rem !important; } /* Centered */

            /* --- Ensure Consistent Centering for Visualization Content --- */
            #visualizations .chart-card {
                margin-left: auto !important;
                margin-right: auto !important;
                margin-bottom: 0rem !important; /* Toolbar will provide space below */
            }

            /* Targets text containers that follow a chart card OR a static-chart-toolbar */
            #visualizations .chart-card + .static-chart-toolbar + .text-container,
            #visualizations .chart-card + .text-container { /* For charts with no toolbar */
                margin-left: auto !important;
                margin-right: auto !important;
                margin-bottom: 2rem !important; /* Standard bottom margin for text blocks */
                margin-top: 0rem !important; /* Text block should sit flush under toolbar or chart */
            }
            /* --- End Consistent Centering --- */

            /* --- MODIFIED Styles for New Static Chart Toolbars --- */
            .static-chart-toolbar {
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
                padding: 0.75rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                background-color: #F8F9FA;
                border-left: 1px solid #dee2e6;
                border-right: 1px solid #dee2e6;
                border-bottom: 1px solid #dee2e6;
                border-top: none;
                border-radius: 0 0 6px 6px;
                display: flex;
                flex-direction: column; /* Stack title and controls-wrapper */
                align-items: flex-start; /* Align title and wrapper to the left */
            }

            .toolbar-main-title {
                font-size: 1rem !important;
                font-weight: 600 !important;
                color: #333 !important;
                margin-bottom: 0.75rem !important;
                text-align: center !important;
                width: 100%; /* Take full width of the toolbar */
            }

            .toolbar-controls-wrapper {
                display: flex;
                flex-direction: row;
                width: 100%;
                align-items: flex-start; /* Align tops of segment/WTP groups */
            }

            .toolbar-control-group {
                display: flex;
                flex-direction: column;
            }
            
            .toolbar-control-group.segment-group {
                flex-grow: 1; /* Allow segment group to take more space if WTP is present */
                 /* If only segment group is present, it will naturally take full width of its parent if parent is 100% */
            }

            .toolbar-control-group.wtp-group {
                min-width: 220px; /* Adjust as needed for WTP options to look good */
                flex-shrink: 0; /* Prevent WTP group from shrinking too much */
            }

            .toolbar-label {
                font-size: 0.9rem !important;
                font-weight: 600 !important; /* Bolder label */
                color: #212529 !important;
                margin-bottom: 0.5rem !important;
                display: block;
                text-align: center !important;
            }

            .toolbar-options-list { /* Container for segment buttons */
                display: flex;
                flex-direction: row;
                flex-wrap: wrap;
                gap: 0.25rem 0.5rem; /* Row and column gap between segment buttons */
                padding-top: 0.25rem; /* Space below label */
            }

            .toolbar-options-grid { /* Container for WTP price points */
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(40px, 1fr)); /* Slightly smaller min for WTP */
                gap: 0.1rem 0.25rem;
                line-height: 1.5;
                padding-top: 0.25rem; /* Space below label */
            }

            .toolbar-segment-option,
            .toolbar-wtp-option {
                font-size: 0.85rem !important;
                color: #6c757d !important;
                padding: 0.25rem 0.5rem !important; /* Uniform padding */
                margin-bottom: 0rem; /* Remove bottom margin, gap handles spacing */
                cursor: pointer !important;
                background-color: transparent !important;
                border: none !important;
                text-align: center !important;
                transition: color 0.2s ease;
                border-radius: 3px;
            }

            .toolbar-wtp-option {
                text-align: center !important;
            }

            .toolbar-segment-option:hover,
            .toolbar-wtp-option:hover {
                color: #212529 !important;
            }

            .toolbar-segment-option.active,
            .toolbar-wtp-option.active {
                color: #212529 !important;
                font-weight: 600 !important;
            }
            
            .toolbar-divider {
                width: 1px;
                background-color: #ccc; /* Or #dee2e6 */
                margin: 0 1rem; /* Spacing around the divider */
                align-self: stretch; /* Make it full height of the controls-wrapper */
            }
            /* --- End Modified Static Toolbar CSS --- */


            .chart-card h4 { color: #1A1A2E !important; font-weight: 600 !important; text-align: center !important; font-size: 1.1rem !important; margin-bottom: 1.25rem !important; }
            #visualizations { margin-bottom: 2rem !important; width: 100%; }

            h3.mb-2 { color: #1A1A2E !important; font-size: 1.85rem !important; font-weight: 700 !important; margin-bottom: 25px !important; position: relative !important; display: inline-block !important; margin-top: 1rem !important; }
            h3.mb-2:after { content: "" !important; position: absolute !important; bottom: -12px !important; left: 0 !important; width: 65px !important; height: 5px !important; background: linear-gradient(90deg, #04C9CE, #5F1AE5) !important; border-radius: 0 !important; }
            .section-header { font-weight: 500 !important; font-size: 1rem !important; color: #2B2B2B !important; }
            .nav-link { font-family: "Inter", sans-serif !important; font-weight: 400 !important; color: #2B2B2B !important; border-radius: 4px !important; margin-bottom: 0.5rem !important; cursor: pointer !important; padding: 0.75rem 1rem !important; font-size: 15px !important; text-decoration: none !important; background: transparent !important; position: relative !important; overflow: hidden !important; z-index: 1 !important; transition: color 0.3s ease-in-out, transform 0.3s ease !important; }
            .carousel-control-prev, .carousel-control-next { display: none !important; }
            #external-carousel-caption { font-weight: bold !important; color: #3E3E3E !important; font-size: 15px !important; position: absolute !important; bottom: 15px !important; left: 1rem !important; right: 1rem !important; text-align: center !important; }
            .nested-links-container { position: relative !important; padding-left: 0 !important; margin-left: 0 !important; margin-top: 0 !important; }
            .nested-links-container::before { content: "" !important; position: absolute !important; left: 1rem !important; top: 0 !important; height: 100% !important; width: 1px !important; background-color: rgba(0, 0, 0, 0.1) !important; z-index: 0 !important; }
            .nav-link.nested-link { padding-left: 2.5rem !important; font-size: 15px !important; margin-bottom: 0.18rem !important; margin-top: 0 !important; position: relative !important; color: #555555 !important; padding-top: 0.35rem !important; padding-bottom: 0.35rem !important; }
            .nested-links-container .nav-link.nested-link:first-child { margin-top: 0.07rem !important; }
            .nested-links-container .nav-link.nested-link:last-child { margin-bottom: 0.07rem !important; }
            .nav-link::after { content: "" !important; position: absolute !important; top: 0 !important; left: 0 !important; right: 0 !important; bottom: 0 !important; background: linear-gradient(135deg, rgba(4, 201, 206, 0.2), rgba(95, 26, 229, 0.2)) !important; opacity: 0 !important; transition: opacity 0.2s ease-out !important; z-index: -1 !important; border-radius: 4px !important; pointer-events: none !important; }
            .nav-link.nested-link::after { left: 1.5rem !important; width: calc(100% - 1.5rem) !important; }
            .nav-link:not(.active):hover::after { opacity: 1 !important; }
            .nav-link.active { background: linear-gradient(135deg, #04C9CE, #5F1AE5) !important; color: white !important; box-shadow: 0 4px 15px rgba(95, 26, 229, 0.2) !important; }
            .nav-link.nested-link.active { padding: 0.35rem 1rem !important; margin: 0 0 0.18rem 1.5rem !important; width: calc(100% - 1.5rem) !important; background: linear-gradient(135deg, #04C9CE, #5F1AE5) !important; border-radius: 4px !important; text-align: left !important; display: flex !important; align-items: center !important; justify-content: flex-start !important; }
            .nav-link.active::before, .nav-link.active::after { display: none !important; }
            .mb-2 { margin-bottom: 0.5rem !important; } .mb-3 { margin-bottom: 1rem !important; } .mb-4 { margin-bottom: 2rem !important; }
            .section-header-card { border: 1px solid #DEE2E6 !important; background-color: #E9ECEF !important; margin-bottom: 0.5rem !important; border-radius: 0.375rem !important; }
            .survey-dual-column { display: flex !important; width: 100% !important; margin: 0 auto !important; } .survey-column { flex: 1 !important; padding: 0 1rem !important; } .survey-divider { width: 1px !important; background-color: #dddddd !important; margin: 0 1rem !important; } .survey-questions, .survey-explanation { font-size: 12px !important; line-height: 1.6 !important; } .survey-dual-column h4 { color: #1A1A2E !important; font-weight: 600 !important; text-align: left !important; font-size: 1rem !important; margin-bottom: 1rem !important; } #survey-questions .chart-card { width: 1100px !important; max-width: 1100px !important; margin: 1.5rem auto 2rem auto !important; }
            .parent-nav-link .nav-link-content { display: flex !important; justify-content: space-between !important; align-items: center !important; width: 100% !important; } .nav-arrow { font-size: 0.8em !important; opacity: 0.6 !important; transition: transform 0.3s ease !important; }
            .pill-button-link { display: block; width: fit-content; margin: 0.75rem auto 0 auto; padding: 0.5rem 1.5rem; border-radius: 50px; text-decoration: none !important; font-weight: 500; font-size: 12px; text-align: center; text-transform: uppercase; cursor: pointer; transition: all 0.2s ease-out; border: 2px solid transparent; background: linear-gradient(white, white) padding-box, linear-gradient(135deg, #04C9CE, #5F1AE5) border-box; -webkit-background-clip: padding-box, border-box; background-clip: padding-box, border-box; color: #2B2B2B !important; box-shadow: none; } .pill-button-link:hover { background: linear-gradient(135deg, #04C9CE, #5F1AE5) padding-box, linear-gradient(135deg, #04C9CE, #5F1AE5) border-box; -webkit-background-clip: padding-box, border-box; background-clip: padding-box, border-box; color: white !important; border-color: transparent; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(95, 26, 229, 0.3); filter: brightness(1.1); } .pill-button-link:active { background: linear-gradient(135deg, #04C9CE, #5F1AE5) padding-box, linear-gradient(135deg, #04C9CE, #5F1AE5) border-box; -webkit-background-clip: padding-box, border-box; background-clip: padding-box, border-box; color: white !important; border-color: transparent; transform: translateY(0px); box-shadow: 0 2px 10px rgba(95, 26, 229, 0.2); filter: brightness(0.95); }
            #carousel-card .carousel-indicators { position: relative; bottom: -25px; z-index: 15; margin-bottom: 0; } #carousel-card .carousel-inner { max-height: 450px; } #carousel-card { padding-bottom: 3rem !important; }
            #carousel-card { width: 85% !important; padding: 1rem !important; padding-bottom: 4rem !important; position: relative !important; max-width: 85% !important; margin-left:auto; margin-right:auto;} /* Applied max-width and centering */
            #carousel-card .carousel-inner { border-radius: 8px !important; overflow: hidden !important; } #carousel-card .carousel-item { text-align: center !important; min-height: 450px; } #carousel-card .carousel-item img { object-fit: contain !important; max-height: 450px !important; width: auto !important; margin: 0 auto !important; } #carousel-card .carousel-indicators { position: absolute !important; bottom: -25px !important; left: 0 !important; right: 0 !important; margin-bottom: 0 !important; z-index: 15 !important; display: flex !important; justify-content: center !important; padding: 0 !important; margin-right: 15% !important; margin-left: 15% !important; list-style: none !important; } #carousel-card .carousel-indicators [data-bs-target], #carousel-card .carousel-indicators li { box-sizing: content-box !important; flex: 0 1 auto !important; width: 30px !important; height: 3px !important; margin-right: 3px !important; margin-left: 3px !important; text-indent: -999px !important; cursor: pointer !important; background-color: #808080 !important; background-clip: padding-box !important; border-top: 10px solid transparent !important; border-bottom: 10px solid transparent !important; opacity: 0.5 !important; transition: opacity 0.6s ease !important; border-radius: 1px !important; } #carousel-card .carousel-indicators [data-bs-target].active, #carousel-card .carousel-indicators li.active { opacity: 1 !important; background-color: #808080 !important; }
            #external-carousel-caption { font-weight: bold !important; color: #3E3E3E !important; font-size: 15px !important; position: absolute !important; bottom: 15px !important; left: 1rem !important; right: 1rem !important; text-align: center !important; }

            /* --- Styles for html.Button segment tabs --- */
            .segment-tab-button {
                background: none !important;
                border: none !important;
                padding: 0.25rem 0.5rem !important; /* Keep original padding */
                margin: 0 !important;
                font: inherit !important; /* Use surrounding font */
                color: #6c757d !important; /* Default gray text */
                cursor: pointer !important;
                outline: inherit !important;
                text-align: center !important;
                border-bottom: 3px solid transparent !important; /* Placeholder */
                transition: color 0.2s ease, border-color 0.2s ease !important;
                /* Add any other styles needed to match the Div appearance */
            }
            .segment-tab-button:hover {
                color: #2B2B2B !important; /* Darken text on hover */
            }
            .segment-tab-button.active {
                color: #2B2B2B !important; /* Black text for active */
                font-weight: 600 !important; /* Bold active text */
                border-bottom-color: transparent !important; /* Keep underline transparent */
            }
            /* --- END: Styles for html.Button segment tabs --- */

            /* --- REVISED: Limitations Carousel Styles --- */
             .limitations-info-container {
                 padding: 0 1rem 0 1rem !important; /* Add horizontal padding */
                 margin-top: 3rem !important; /* Spacing after the main limitations text */
                 margin-bottom: 2rem !important;
                 max-width: 900px !important; /* Match other content cards */
                 margin-left: auto !important; /* Center */
                 margin-right: auto !important; /* Center */
                 border: 1px solid #DEE2E6 !important; /* Match segment container border */
                 min-height: 650px !important; /* INCREASED minimum height significantly */
                 height: 650px !important; /* Fixed height like segment container */
                 display: flex !important;
                 flex-direction: column !important;
                 background-color: white !important;
                 border-radius: 12px !important; /* Match segment container radius */
                 box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; /* Match segment container shadow */
                 overflow: hidden; /* Needed for border-radius on tabs */
             }

            #limitation-title-display { /* Styling for the dynamic title */
                font-weight: bold !important;
                margin-bottom: 1rem !important;
                padding-left: 0 !important; /* Remove specific padding if container has padding */
                padding-top: 1.5rem !important; /* Add padding top to space it */
                flex-shrink: 0; /* Prevent title from shrinking */
            }

            #limitation-content-display {
                padding: 0 0 1.5rem 0 !important; /* Adjust padding (removed horizontal) */
                flex-grow: 1; /* Allow content area to grow */
                width: 100% !important;
                overflow-y: auto !important; /* Allow content to scroll if needed */
                /* Removed height:0 and calc() - rely on flex-grow and container height */
             }

            .limitation-selector-tabs {
                display: flex !important;
                justify-content: flex-start !important; /* Align tabs to the start */
                align-items: stretch; /* Make buttons same height if wrapped */
                border-top: 1px solid #DEE2E6 !important;
                padding: 0.75rem 0rem !important; /* Adjust padding */
                margin-top: auto !important; /* Push tabs to the bottom */
                flex: 0 0 auto !important; /* Prevent tabs area from growing */
                background-color: white !important; /* Match container background */
                width: 100% !important;
                flex-wrap: wrap !important; /* ** Allow tabs to wrap ** */
                gap: 0.5rem 1rem !important; /* Add vertical and horizontal gap between wrapped tabs */
                 border-bottom-left-radius: 12px !important; /* Match container radius */
                 border-bottom-right-radius: 12px !important; /* Match container radius */
            }

            .limitation-tab-button {
                background: none !important;
                border: none !important;
                padding: 0.25rem 0.5rem !important; /* Match segment tab padding */
                margin: 0 !important;
                font: inherit !important;
                font-family: "Inter", sans-serif !important; /* Ensure consistent font */
                color: #6c757d !important; /* Default gray text */
                cursor: pointer !important;
                outline: inherit !important;
                text-align: center !important;
                border-bottom: 3px solid transparent !important; /* Transparent bottom border */
                transition: color 0.2s ease !important; /* Simplify transition */
                font-size: 0.85rem !important; /* Keep slightly smaller font */
                line-height: 1.3 !important;
                flex-grow: 0;
                flex-shrink: 0;
                flex-basis: auto;
                white-space: normal !important; /* Allow text wrapping */
                border-radius: 0 !important; /* No button radius */
            }

            .limitation-tab-button:hover {
                color: #2B2B2B !important; /* Darken text on hover */
                background-color: transparent !important; /* NO background on hover */
            }

            .limitation-tab-button.active {
                color: #2B2B2B !important; /* Black text for active */
                font-weight: 600 !important; /* Bold active text */
                border-bottom-color: transparent !important; /* NO bottom border for active */
                background-color: transparent !important; /* NO background for active */
            }
            /* --- END: Limitations Carousel Styles --- */

        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
<script>
         // --- Start of Script Block ---
         console.log('>>> Executing index_string script block <<<');

         // Encapsulate all DOM-dependent initialization logic
         function initializeDashboardScripting() {
            console.log('>>> initializeDashboardScripting: Attempting to find core elements...');
            const navLinks = document.querySelectorAll('.dashboard-sidebar .nav-link');
            const scrollContainer = document.getElementById('page-content');

            if (navLinks.length === 0 || !scrollContainer) {
                console.warn('>>> initializeDashboardScripting: Core elements (navLinks or scrollContainer) not found yet. navLinks count:', navLinks.length, 'scrollContainer:', scrollContainer);
                return false; 
            }
            console.log('>>> initializeDashboardScripting: Core elements found. navLinks count:', navLinks.length, 'scrollContainer ID:', scrollContainer.id);
            console.log('>>> Proceeding with full script initialization...');

            function setupArrowObserver(collapseId, navLinkId) {
                const collapseElement = document.getElementById(collapseId);
                const navLinkElement = document.getElementById(navLinkId);
                const arrowElement = navLinkElement ? navLinkElement.querySelector('.nav-arrow') : null;

                if (!collapseElement || !arrowElement || !navLinkElement) {
                    return;
                }

                const observer = new MutationObserver(mutations => {
                    mutations.forEach(mutation => {
                        if (mutation.attributeName === 'class') {
                            const isShown = collapseElement.classList.contains('show');
                            arrowElement.style.transform = isShown ? 'rotate(180deg)' : 'rotate(0deg)';
                        }
                    });
                });
                observer.observe(collapseElement, { attributes: true });
                const isInitiallyShown = collapseElement.classList.contains('show');
                arrowElement.style.transform = isInitiallyShown ? 'rotate(180deg)' : 'rotate(0deg)';
            }

            setupArrowObserver('collapse-overview', 'nav-overview');
            setupArrowObserver('collapse-methodology', 'nav-methodology-design');
            setupArrowObserver('collapse-my-role', 'nav-my-role');
            setupArrowObserver('collapse-visualizations', 'nav-visualizations');
            
            window.clickNavInProgress = false; 

            navLinks.forEach(function(link) {
               console.log('>>> Attaching listener to:', link.id || link.href);
               link.addEventListener('click', function(e) {
                    console.log('>>> Click event FIRED for:', this.id || this.href);
                    const href = this.getAttribute('href');
                    console.log('>>> Click handler: Got href:', href);

                    if (href && href.startsWith('#')) {
                        console.log('>>> Click handler: Preventing default for hash link.');
                        e.preventDefault();

                        const isParent = this.classList.contains('parent-nav-link');
                        console.log('>>> Click handler: Is parent link?', isParent);

                        const targetId = href.substring(1);
                        const targetElement = document.getElementById(targetId);

                        console.log('>>> Click handler: Checking elements: targetId=', targetId, ' targetElement=', targetElement, ' scrollContainer=', scrollContainer);

                        if (isParent && targetElement) {
                            console.log('>>> Click handler: Entering PARENT link scroll logic.');
                            window.clickNavInProgress = true;
                            const containerRect = scrollContainer.getBoundingClientRect();
                            const targetRect = targetElement.getBoundingClientRect();
                            const scrollPosition = scrollContainer.scrollTop;
                            const targetTopRelativeToContainer = targetRect.top - containerRect.top;
                            const offset = 20;
                            let scrollTo = scrollPosition + targetTopRelativeToContainer - offset;
                            scrollTo = Math.max(0, scrollTo);

                            console.log('Parent NavLink - Attempting to scroll to:', scrollTo, 'for target:', targetId);
                            scrollContainer.scrollTo({ top: scrollTo, behavior: 'auto' }); 

                            if (history.pushState) { history.pushState(null, null, href); }
                            setTimeout(() => { window.clickNavInProgress = false; }, 150); 
                            return;
                        } else if (isParent) {
                            console.log('>>> Click handler: Entering PARENT link NO SCROLL logic (just hash update).');
                            if (history.pushState) { history.pushState(null, null, href); }
                            return; 
                        } else if (targetElement && scrollContainer) {
                            console.log('>>> Click handler: Entering NESTED link scroll logic.');
                            window.clickNavInProgress = true;
                            const containerRect = scrollContainer.getBoundingClientRect();
                            const targetRect = targetElement.getBoundingClientRect();
                            const scrollPosition = scrollContainer.scrollTop;
                            const targetTopRelativeToContainer = targetRect.top - containerRect.top;
                            let scrollTo;
                            const offset = 20;

                            if (targetId.includes('-chart-container') || targetId === 'limitations-info-container' || targetId === 'segment-info-container') {
                                const targetHeight = targetRect.height;
                                const containerHeight = containerRect.height;
                                scrollTo = scrollPosition + targetTopRelativeToContainer - (containerHeight / 2) + (targetHeight / 2);
                                scrollTo = Math.max(0, scrollTo);
                                scrollTo = Math.min(scrollContainer.scrollHeight - containerHeight, scrollTo);
                            } else {
                                scrollTo = scrollPosition + targetTopRelativeToContainer - offset;
                                scrollTo = Math.max(0, scrollTo);
                            }
                            console.log('Nested NavLink - Attempting to scroll to:', scrollTo, 'for target:', targetId);
                            scrollContainer.scrollTo({ top: scrollTo, behavior: 'auto' }); 

                            if (history.pushState) { history.pushState(null, null, href); }
                            setTimeout(() => { window.clickNavInProgress = false; }, 150); 
                        } else {
                            console.warn('>>> Click handler: NESTED link scroll skipped. targetElement=', targetElement, ' scrollContainer=', scrollContainer);
                        }
                    } else {
                       console.log('>>> Click handler: Ignoring click on non-hash link or link without href:', this.id);
                    }
               });
            });
            
            const initialHash = window.location.hash;
            if (initialHash && initialHash !== '#') {
                const targetId = initialHash.substring(1);
                console.log(">>> Initial hash detected:", initialHash);
                const scrollToTarget = () => {
                    console.log(">>> scrollToTarget function executing for:", targetId);
                    try {
                        const targetElement = document.getElementById(targetId);
                        console.log('Initial Scroll - Target:', targetElement ? targetElement.id : 'null', 'Scroll Container:', scrollContainer ? 'found' : 'null');

                        if (!scrollContainer || !targetElement) {
                            console.warn('Initial scroll aborted: target or container missing for', targetId);
                            return;
                        }
                        const containerRect = scrollContainer.getBoundingClientRect();
                        const targetRect = targetElement.getBoundingClientRect();
                        const scrollPosition = scrollContainer.scrollTop;
                        const targetTopRelativeToContainer = targetRect.top - containerRect.top;
                        let scrollTo;
                        const offset = 20;

                        if (targetId.includes('-chart-container') || targetId === 'limitations-info-container' || targetId === 'segment-info-container') {
                            const targetHeight = targetRect.height;
                            const containerHeight = containerRect.height;
                            scrollTo = scrollPosition + targetTopRelativeToContainer - (containerHeight / 2) + (targetHeight / 2);
                            scrollTo = Math.max(0, scrollTo);
                            scrollTo = Math.min(scrollContainer.scrollHeight - containerHeight, scrollTo);
                        } else {
                            scrollTo = scrollPosition + targetTopRelativeToContainer - offset;
                            scrollTo = Math.max(0, scrollTo);
                        }
                        console.log('Initial scrolling to:', scrollTo, 'for target:', targetId);
                        scrollContainer.scrollTo({ top: scrollTo, behavior: 'auto' });
                    } catch (e) { console.error("Error during initial scroll:", e); }
                };

                const targetElementForInitialScroll = document.getElementById(targetId);
                if (targetElementForInitialScroll) {
                    console.log(">>> Initial scroll: Target element found:", targetId);
                    const parentCollapse = targetElementForInitialScroll.closest('.collapse:not(.show)');
                    let needsExpand = false;
                    let controllingNavLinkId = null;

                    if(parentCollapse){
                        console.log(">>> Initial scroll: Target is inside a collapsed section:", parentCollapse.id);
                        const collapseId = parentCollapse.id;
                        const navLinkForCollapse = Array.from(navLinks).find(link => {
                            const linkHref = link.getAttribute('href');
                            if (linkHref && linkHref.substring(1) === collapseId.replace('collapse-','')) {
                                return true;
                            }
                            return false;
                        });
                        if (navLinkForCollapse && navLinkForCollapse.classList.contains('parent-nav-link')) {
                            controllingNavLinkId = navLinkForCollapse.id;
                            needsExpand = true;
                            console.log(">>> Initial scroll: Found controlling parent nav link:", controllingNavLinkId, "for collapse:", collapseId);
                        } else {
                           console.warn(">>> Initial scroll: Could not find controlling parent nav link for collapse:", collapseId);
                        }
                    }

                    if (needsExpand && controllingNavLinkId) {
                        console.log(">>> Initial scroll: Needs expand. Clicking parent link and delaying scroll.");
                        const parentNavLink = document.getElementById(controllingNavLinkId);
                        if (parentNavLink) {
                            parentNavLink.click(); 
                            setTimeout(scrollToTarget, 450);
    } else {
                            console.warn(">>> Initial scroll: Controlling nav link element not found by ID:", controllingNavLinkId);
                            setTimeout(scrollToTarget, 150);
                        }
                    } else {
                        console.log(">>> Initial scroll: No expand needed or target not in collapsed section. Scrolling directly.");
                        setTimeout(scrollToTarget, 150);
                    }
                } else {
                    console.warn(">>> Initial scroll: Target element not found for hash:", initialHash);
                }
            }
            // --- End Initial Hash Scroll ---

            console.log('>>> initializeDashboardScripting: Successfully set up all event listeners and logic.');
            return true; 
         } 


         window.addEventListener('load', function() {
            console.log('>>> Load event listener fired <<<');

            let attempts = 0;
            const maxAttempts = 30; 
            const intervalTime = 500; 

            function attemptInitialization() {
                console.log(`>>> attemptInitialization: Attempt ${attempts + 1}/${maxAttempts}`);
                if (initializeDashboardScripting()) {
                    console.log('>>> attemptInitialization: Dashboard scripting initialized successfully.');
                } else {
                    attempts++;
                    if (attempts < maxAttempts) {
                        console.log(`>>> attemptInitialization: Retrying in ${intervalTime}ms...`);
                        setTimeout(attemptInitialization, intervalTime);
                    } else {
                        console.error('>>> attemptInitialization: Max attempts reached. Failed to initialize dashboard scripting. Some interactive features might not work.');
                    }
                }
            }

            attemptInitialization(); 

            console.log('>>> Load event listener processing finished. <<<');
         });
        </script>
    </body>
</html>
'''

# --- Callback to update Segment Info Display ---
@app.callback(
    Output('segment-content-display', 'children'),
    Output('segment-title-display', 'children'), # NEW Output for title
    Output({'type': 'segment-tab', 'index': ALL}, 'className'),
    Input({'type': 'segment-tab', 'index': ALL}, 'n_clicks'),
    State({'type': 'segment-tab', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def update_segment_info(n_clicks_list, tab_ids):
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate
    if isinstance(ctx.triggered_id, str):
        raise dash.exceptions.PreventUpdate
    clicked_segment = ctx.triggered_id.get('index')
    if not clicked_segment:
        raise dash.exceptions.PreventUpdate

    # Get content AND title
    new_content = create_segment_content(clicked_segment)
    new_title = SEGMENT_INFO.get(clicked_segment, {}).get("title", "Segment Details") # Get title from SEGMENT_INFO

    new_class_names = [
        'segment-tab-button active' if tab_id['index'] == clicked_segment else 'segment-tab-button'
        for tab_id in tab_ids
    ]
    return new_content, new_title, new_class_names # Return title as well

# --- END: Segment Callback ---

# --- REVISED Callback to update Limitation Info Display ---
@app.callback(
    Output('limitation-content-display', 'children'),
    Output('limitation-title-display', 'children'), # NEW Output for the title H4
    Output({'type': 'limitation-tab', 'index': ALL}, 'className'),
    Input({'type': 'limitation-tab', 'index': ALL}, 'n_clicks'),
    State({'type': 'limitation-tab', 'index': ALL}, 'id'),
    State('limitation-content-store', 'data'), # Get content from the store
    prevent_initial_call=True
)
def update_limitation_info(n_clicks_list, tab_ids, stored_limitations_content):
    if not ctx.triggered_id or not stored_limitations_content:
        raise dash.exceptions.PreventUpdate
    if isinstance(ctx.triggered_id, str):
         raise dash.exceptions.PreventUpdate

    clicked_limitation_id = ctx.triggered_id.get('index')
    if not clicked_limitation_id:
        raise dash.exceptions.PreventUpdate

    # Generate content for the clicked tab
    new_content = create_limitation_content(clicked_limitation_id, stored_limitations_content)

    # Find the title for the clicked limitation ID
    new_title = "Limitation Details" # Default title
    for item in stored_limitations_content:
        if item['id'] == clicked_limitation_id:
            new_title = item['title']
            break

    # Update class names for all tabs: set the clicked one 'active' (using revised styling)
    new_class_names = [
        f"limitation-tab-button {'active' if tab_id['index'] == clicked_limitation_id else ''}"
        for tab_id in tab_ids
    ]

    return new_content, new_title, new_class_names # Return content, title, and class names
# --- END: Revised Limitation Callback ---


# --- Populate Limitation Content Store on Load ---
@app.callback(Output('limitation-content-store', 'data'), Input('url', 'pathname'))
def store_limitations_content_on_load(pathname):
    print("Storing limitations content...")
    return limitations_content
# --- END: Populate Limitation Store ---

@app.callback(Output('selected-values-store', 'data'), Input('url', 'pathname'))
def init_selected_values(pathname):
     print("Initializing selected-values-store...")
     return {'segment': 'All Segments', 'wtp': 139}


@app.callback(
    [
        Output({'type': 'static-segment-options', 'chart': ALL}, 'children'),
        Output({'type': 'static-wtp-options', 'chart': ALL}, 'children')
    ],
    Input('url', 'pathname')
)
def populate_static_toolbar_options(pathname):
    if app_data is None:
        no_data_message = "Data not available."
        num_segment_outputs = len(ctx.outputs_list[0])
        num_wtp_outputs = len(ctx.outputs_list[1])
        return [[no_data_message]] * num_segment_outputs, [[no_data_message]] * num_wtp_outputs

    # Define which charts get which controls
    charts_with_segment_options = ['regional-map', 'expansion-matrix', 'vw-control', 'vw-test', 'wtp-gg']
    charts_with_wtp_options = ['regional-map', 'expansion-matrix']

    segments = ['All Segments'] + sorted(app_data['Segment'].unique())
    wtp_prices = [79, 99, 119, 139, 159, 179, 199]
    default_segment = 'All Segments'
    default_wtp = 139

    segment_options_outputs = [dash.no_update] * len(ctx.outputs_list[0])
    wtp_options_outputs = [dash.no_update] * len(ctx.outputs_list[1])

    # Populate segment options
    for i, output_id_obj in enumerate(ctx.outputs_list[0]):
        chart_id = output_id_obj['id']['chart']
        if chart_id in charts_with_segment_options:
            options = [
                html.Div(
                    segment_name,
                    id={'type': 'toolbar-segment-button', 'chart': chart_id, 'index': segment_name},
                    className="toolbar-segment-option" + (" active" if segment_name == default_segment else "")
                ) for segment_name in segments
            ]
            segment_options_outputs[i] = options

    # Populate WTP options
    for i, output_id_obj in enumerate(ctx.outputs_list[1]):
        chart_id = output_id_obj['id']['chart']
        if chart_id in charts_with_wtp_options:
            options = [
                html.Div(
                    f"${price}",
                    id={'type': 'toolbar-wtp-button', 'chart': chart_id, 'index': price},
                    className="toolbar-wtp-option" + (" active" if price == default_wtp else "")
                ) for price in wtp_prices
            ]
            wtp_options_outputs[i] = options
            
    return segment_options_outputs, wtp_options_outputs


@app.callback(
    Output('selected-values-store', 'data', allow_duplicate=True),
    Output({'type': 'toolbar-segment-button', 'chart': ALL, 'index': ALL}, 'className'),
    Output({'type': 'toolbar-wtp-button', 'chart': ALL, 'index': ALL}, 'className'),
    Input({'type': 'toolbar-segment-button', 'chart': ALL, 'index': ALL}, 'n_clicks'),
    Input({'type': 'toolbar-wtp-button', 'chart': ALL, 'index': ALL}, 'n_clicks'),
    State('selected-values-store', 'data'),
    State({'type': 'toolbar-segment-button', 'chart': ALL, 'index': ALL}, 'id'),
    State({'type': 'toolbar-wtp-button', 'chart': ALL, 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def update_selected_values_from_static_toolbars(segment_clicks, wtp_clicks, current_values, segment_ids, wtp_ids):
    triggered_id_obj = ctx.triggered_id
    print(f"Static Toolbar update triggered by: {triggered_id_obj}")

    if not triggered_id_obj:
        raise dash.exceptions.PreventUpdate
    
    # Check if triggered_id_obj is a string (can happen on initial load if not handled carefully)
    if isinstance(triggered_id_obj, str):
        print(f"Static Toolbar update triggered by string ID: {triggered_id_obj}, preventing update.")
        raise dash.exceptions.PreventUpdate

    current_segment = current_values.get('segment', 'All Segments')
    current_wtp = current_values.get('wtp', 139)
    new_values = current_values.copy()

    clicked_item_type = triggered_id_obj.get('type')
    clicked_item_index = triggered_id_obj.get('index') # This is the value (e.g., 'Curious Skeptics' or 139)
    # clicked_item_chart = triggered_id_obj.get('chart') # Chart specific to the button clicked

    if clicked_item_type == 'toolbar-segment-button' and clicked_item_index is not None:
        new_values['segment'] = clicked_item_index
        print(f"Segment selected: {new_values['segment']}")
    elif clicked_item_type == 'toolbar-wtp-button' and clicked_item_index is not None:
        new_values['wtp'] = clicked_item_index
        print(f"WTP selected: {new_values['wtp']}")
    else:
        print("No relevant toolbar button was clicked or index is missing.")
        raise dash.exceptions.PreventUpdate

    segment_button_class_names = []
    for seg_id_obj in segment_ids:
        # seg_id_obj is {'type': 'toolbar-segment-button', 'chart': 'chart_name', 'index': 'segment_value'}
        current_button_segment_value = seg_id_obj['index']
        base_class = "toolbar-segment-option"
        if current_button_segment_value == new_values['segment']:
            segment_button_class_names.append(f"{base_class} active")
        else:
            segment_button_class_names.append(base_class)

    wtp_button_class_names = []
    for wtp_id_obj in wtp_ids:
        # wtp_id_obj is {'type': 'toolbar-wtp-button', 'chart': 'chart_name', 'index': price_value}
        current_button_wtp_value = wtp_id_obj['index']
        base_class = "toolbar-wtp-option"
        if current_button_wtp_value == new_values['wtp']:
            wtp_button_class_names.append(f"{base_class} active")
        else:
            wtp_button_class_names.append(base_class)
            
    return new_values, segment_button_class_names, wtp_button_class_names


# --- Run the App Server ---
if __name__ == '__main__':
    try:
        import statsmodels
        import sklearn
    except ImportError as e:
        print(f"\n*** DEPENDENCY ERROR: {e}. Please install required libraries: ***")
        print("*** pip install statsmodels scikit-learn ***\n")
        exit()

    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("\n*** ACTION NEEDED: Created 'assets' folder. ***")
        print(f"*** Please move your images ('final flowchart.png', 'conjoint_rejection_discussion.png', etc.) to this folder. ***\n")

    if app_data is None:
        print(f"\n*** WARNING: Global data loading failed (or file '{DATA_FILE}' not found/unreadable). Dashboard will run with limited functionality. ***\n")
    else:
        print(f"Global data from '{DATA_FILE}' successfully loaded. Starting server...")

    app.run_server(debug=True, port=8051)