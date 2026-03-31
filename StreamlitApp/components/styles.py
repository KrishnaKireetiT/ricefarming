"""
Global CSS styles for the Pipeline Testing App.
Provides a modern, polished look across both Technical and Extension Officer modes.
"""

GLOBAL_CSS = """
<style>
/* ============================================ */
/* Global Styles                                */
/* ============================================ */

/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base typography */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Ensure enough top padding so tabs aren't clipped */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* Collapse the default Streamlit top decoration bar */
header[data-testid="stHeader"] {
    height: 2rem !important;
    min-height: 2rem !important;
    background: transparent !important;
}

/* ============================================ */
/* Toggle Switches                              */
/* ============================================ */

/* Style the toggle container in sidebar */
[data-testid="stSidebar"] .stToggle > label {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Green track when on */
[data-testid="stSidebar"] .stToggle > label > div[data-checked="true"] > div {
    background-color: #10b981 !important;
}

/* ============================================ */
/* Sidebar Polish                               */
/* ============================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafbfc 0%, #f0f2f6 100%);
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Mode switch styling */
.mode-switch-container {
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
    margin: 8px 0;
}

/* ============================================ */
/* Tab Bar                                      */
/* ============================================ */

/* Radio button tabs - make them look like proper tabs */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stRadio > div {
    gap: 0 !important;
}

.stRadio > div[role="radiogroup"] {
    gap: 4px !important;
}

.stRadio > div[role="radiogroup"] label {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 20px !important;
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 0.9rem;
}

.stRadio > div[role="radiogroup"] label:hover {
    background: #ecfdf5;
    border-color: #34d399;
}

.stRadio > div[role="radiogroup"] label[data-checked="true"],
.stRadio > div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white !important;
    border-color: transparent;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

/* ============================================ */
/* Metric Cards                                 */
/* ============================================ */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border-color: #cbd5e1;
}

[data-testid="stMetricLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b !important;
}

[data-testid="stMetricValue"] div {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

[data-testid="stMetricDelta"] div {
    font-size: 0.85rem !important;
}

/* ============================================ */
/* Buttons                                      */
/* ============================================ */

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 24px !important;
    transition: all 0.2s ease !important;
    border: 1px solid #e2e8f0 !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4) !important;
}

/* ============================================ */
/* Text Areas & Inputs                          */
/* ============================================ */

.stTextArea textarea,
.stTextInput input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
}

/* ============================================ */
/* Expanders                                    */
/* ============================================ */

.streamlit-expanderHeader {
    background: #f8fafc !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    border: 1px solid #e2e8f0 !important;
}

.streamlit-expanderHeader:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
}

details[open] > summary.streamlit-expanderHeader {
    border-radius: 10px 10px 0 0 !important;
}

.streamlit-expanderContent {
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 16px !important;
}

/* ============================================ */
/* Dividers                                     */
/* ============================================ */

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ============================================ */
/* Select Boxes                                 */
/* ============================================ */

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
}

/* ============================================ */
/* Forms                                        */
/* ============================================ */

[data-testid="stForm"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px !important;
}

/* ============================================ */
/* Sliders                                      */
/* ============================================ */

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #e2e8f0, #10b981) !important;
}

.stSlider [data-testid="stThumbValue"] {
    font-weight: 700 !important;
    color: #059669 !important;
}

/* ============================================ */
/* Progress Bar                                 */
/* ============================================ */

.stProgress > div > div > div {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    border-radius: 8px !important;
}

/* ============================================ */
/* Alerts / Info / Success / Warning             */
/* ============================================ */

.stAlert {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ============================================ */
/* File Uploader                                */
/* ============================================ */

[data-testid="stFileUploader"] {
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] section {
    border-radius: 12px !important;
    border: 2px dashed #cbd5e1 !important;
    padding: 24px !important;
}

/* ============================================ */
/* Download Buttons                             */
/* ============================================ */

.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1.5px solid #e2e8f0 !important;
}

.stDownloadButton > button:hover {
    background: #f1f5f9 !important;
    border-color: #10b981 !important;
}

/* ============================================ */
/* Custom Classes                               */
/* ============================================ */

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    padding: 24px;
    border-radius: 14px;
    border-left: 5px solid #10b981;
    margin: 12px 0;
    line-height: 1.7;
    font-size: 0.95rem;
    color: #1e293b;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
}

/* Pipeline answer boxes for side-by-side */
.pipeline-answer-a {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 20px 24px;
    border-radius: 14px;
    border-left: 5px solid #3b82f6;
    min-height: 200px;
    line-height: 1.7;
    font-size: 0.95rem;
    color: #1e293b;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08);
}

.pipeline-answer-b {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    padding: 20px 24px;
    border-radius: 14px;
    border-left: 5px solid #10b981;
    min-height: 200px;
    line-height: 1.7;
    font-size: 0.95rem;
    color: #1e293b;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
}

/* Info cards (for retrieval details, entity panels) */
.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin: 8px 0;
}

/* Entity chips */
.entity-chip {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px 4px;
    border: 1px solid #c7d2fe;
}

.entity-chip-aligned {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px 4px;
    border: 1px solid #a7f3d0;
}

/* Section headers with icon */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin: 16px 0 12px 0;
}

/* Score badge */
.score-badge {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'SF Mono', 'Cascadia Code', monospace;
}

.score-badge-high {
    background: #dcfce7;
    color: #166534;
}

.score-badge-medium {
    background: #fef9c3;
    color: #854d0e;
}

.score-badge-low {
    background: #fee2e2;
    color: #991b1b;
}

/* Preference badge */
.pref-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.pref-badge-a {
    background: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #93c5fd;
}

.pref-badge-b {
    background: #d1fae5;
    color: #047857;
    border: 1px solid #6ee7b7;
}

.pref-badge-tie {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}

/* History card refined */
.history-card {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    margin-bottom: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

.history-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    border-color: #93c5fd !important;
    transform: translateY(-1px);
}

.card-question {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-bottom: 10px !important;
    line-height: 1.5 !important;
}

.card-answer-preview {
    color: #475569 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    margin-bottom: 12px !important;
    padding: 10px 14px !important;
    background: rgba(16, 185, 129, 0.06) !important;
    border-left: 3px solid #10b981 !important;
    border-radius: 6px !important;
}

/* Retrieval source badges */
.retrieval-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
    color: white;
}

.retrieval-badge-kg { background: #059669; }
.retrieval-badge-semantic { background: #2563eb; }
.retrieval-badge-keyword { background: #d97706; }
.retrieval-badge-structured { background: #7c3aed; }

/* Fact display */
.fact-row {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    font-size: 0.9rem;
    transition: background 0.15s;
}

.fact-row:hover {
    background: #f1f5f9;
}

/* Chunk display card */
.chunk-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.chunk-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #94a3b8;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 12px;
}

.empty-state-text {
    font-size: 1rem;
    font-weight: 500;
}
</style>
"""

DARK_THEME_CSS = """
<style>
/* ============================================ */
/* Dark Theme Overrides                         */
/* ============================================ */

:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --border-color: #334155;
    --accent: #34d399;
}

.stApp {
    background-color: var(--bg-primary);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
    color: var(--text-secondary);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

.stMarkdown, p, span, label {
    color: var(--text-secondary);
}

/* Metric cards dark */
[data-testid="stMetric"] {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] div {
    color: var(--accent) !important;
}

/* Buttons dark */
.stButton > button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

.stButton > button:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--accent) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
}

/* Inputs dark */
.stTextArea textarea,
.stTextInput input {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.2) !important;
}

/* Expanders dark */
.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderContent {
    background: var(--bg-tertiary) !important;
    border-color: var(--border-color) !important;
}

/* Select box dark */
.stSelectbox > div > div {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

/* Radio dark */
.stRadio > div[role="radiogroup"] label {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
    color: var(--text-secondary) !important;
}

.stRadio > div[role="radiogroup"] label:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--accent) !important;
}

/* Tab bar active in dark */
.stRadio > div[role="radiogroup"] label[data-checked="true"],
.stRadio > div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border-color: transparent !important;
}

/* Answer boxes dark */
.answer-box {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%) !important;
    color: #d1fae5 !important;
    border-left-color: #34d399 !important;
}

.pipeline-answer-a {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e40af33 100%) !important;
    color: #dbeafe !important;
    border-left-color: #60a5fa !important;
}

.pipeline-answer-b {
    background: linear-gradient(135deg, #064e3b 0%, #065f4633 100%) !important;
    color: #d1fae5 !important;
    border-left-color: #34d399 !important;
}

/* Info card dark */
.info-card {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* Entity chips dark */
.entity-chip {
    background: #312e81 !important;
    color: #a5b4fc !important;
    border-color: #4338ca !important;
}

.entity-chip-aligned {
    background: #064e3b !important;
    color: #6ee7b7 !important;
    border-color: #059669 !important;
}

/* Fact rows dark */
.fact-row {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

.fact-row:hover {
    background: var(--bg-tertiary) !important;
}

/* Chunk cards dark */
.chunk-card {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* History card dark */
.history-card {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

.history-card:hover {
    border-color: var(--accent) !important;
}

.card-question {
    color: var(--text-primary) !important;
}

.card-answer-preview {
    color: var(--text-secondary) !important;
    background: rgba(16, 185, 129, 0.1) !important;
}

/* Forms dark */
[data-testid="stForm"] {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* Alerts dark */
.stAlert {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* Divider dark */
hr {
    background: linear-gradient(90deg, transparent, var(--border-color) 20%, var(--border-color) 80%, transparent) !important;
}

/* Code blocks dark */
code {
    background: var(--bg-tertiary) !important;
    color: var(--accent) !important;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Tables dark */
.stDataFrame {
    background: var(--bg-secondary) !important;
}

/* Links */
a {
    color: var(--accent) !important;
}
a:hover {
    color: #a5b4fc !important;
}

/* Blockquotes dark */
blockquote {
    border-left-color: var(--accent) !important;
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    padding: 8px 16px;
    border-radius: 0 8px 8px 0;
}

/* ============================================ */
/* Dark: ALL text visibility fixes              */
/* ============================================ */

/* Section headers (custom HTML) */
.section-header {
    color: var(--text-primary) !important;
}

/* Score badges dark */
.score-badge {
    background: var(--bg-tertiary) !important;
    color: var(--text-secondary) !important;
}
.score-badge-high {
    background: #064e3b !important;
    color: #6ee7b7 !important;
}
.score-badge-medium {
    background: #713f12 !important;
    color: #fde68a !important;
}
.score-badge-low {
    background: #7f1d1d !important;
    color: #fca5a5 !important;
}

/* Preference badges dark */
.pref-badge-a {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border-color: #3b82f6 !important;
}
.pref-badge-b {
    background: #064e3b !important;
    color: #6ee7b7 !important;
    border-color: #10b981 !important;
}
.pref-badge-tie {
    background: var(--bg-tertiary) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-color) !important;
}

/* Retrieval badges - keep colored backgrounds, ensure white text */
.retrieval-badge {
    color: white !important;
}

/* Captions */
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
}

/* Sidebar text overrides */
[data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
    color: var(--text-secondary);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: var(--text-primary) !important;
}

/* Sidebar toggle label */
[data-testid="stSidebar"] .stToggle label span {
    color: var(--text-secondary) !important;
}

/* Empty state dark */
.empty-state {
    color: var(--text-muted) !important;
}

/* Select dropdown dark */
[data-testid="stSelectbox"] label,
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
}

/* Multiselect / checkbox dark */
.stCheckbox label span, .stMultiSelect label {
    color: var(--text-secondary) !important;
}

/* Status widget dark */
[data-testid="stStatusWidget"] {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
}

/* Toast / success / warning / error text */
.stAlert p, .stAlert span, .stAlert div {
    color: inherit !important;
}

/* File uploader dark */
[data-testid="stFileUploader"] section {
    border-color: var(--border-color) !important;
    background: var(--bg-secondary) !important;
}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {
    color: var(--text-secondary) !important;
}

/* Download button dark */
.stDownloadButton > button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

/* Slider labels dark */
.stSlider label, .stSlider p, .stSlider span {
    color: var(--text-secondary) !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: var(--accent) !important;
}

/* Number input dark */
.stNumberInput label {
    color: var(--text-secondary) !important;
}
.stNumberInput input {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

/* Tabs (native st.tabs) dark */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-secondary) !important;
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    background-color: var(--bg-tertiary) !important;
}

/* Ensure pipeline answer text is always readable */
.pipeline-answer-a, .pipeline-answer-a p,
.pipeline-answer-b, .pipeline-answer-b p,
.answer-box, .answer-box p {
    color: inherit !important;
}

/* Theme button styling */
button[key="theme_btn"], [data-testid="baseButton-secondary"] {
    font-size: 1.2rem !important;
    padding: 4px 8px !important;
    min-height: 2.2rem !important;
}
</style>
"""
