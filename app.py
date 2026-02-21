import json
from pathlib import Path
import streamlit as st

from generator import generate_from_config, save_and_zip

st.set_page_config(page_title="AP Demo Generator", layout="wide")

APP_TITLE = "AP Demo Data Generator"
SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(exist_ok=True)

st.title(APP_TITLE)
st.caption("Choose a scenario or paste/upload Lovable JSON → Generate relational CSVs + KPIs + ZIP")

# -----------------------
# Sample scenario library
# -----------------------
SAMPLE_SCENARIOS = {
    "Baseline (All Pass)": [
        {"name": "Vendors", "recordCount": 50, "isDependent": False, "fields": []},
        {"name": "Line Items", "recordCount": 100, "isDependent": False, "fields": []},
        {"name": "POs", "recordCount": 50, "isDependent": True, "fields": []},
        {"name": "Contracts", "recordCount": 10, "isDependent": True, "fields": []},
        {"name": "Invoices", "recordCount": 100, "isDependent": True, "fields": [
            {"name": "PO_MATCHED", "type": "enum", "distribution": [{"label": "PO_MATCHED", "percentage": 100}]},
            {"name": "PRICE_VARIENCE", "type": "enum", "distribution": [{"label": "NONE", "percentage": 100}]},
        ]},
    ],
    "Missing PO (10%)": [
        {"name": "Vendors", "recordCount": 50, "isDependent": False, "fields": []},
        {"name": "Line Items", "recordCount": 100, "isDependent": False, "fields": []},
        {"name": "POs", "recordCount": 50, "isDependent": True, "fields": []},
        {"name": "Contracts", "recordCount": 10, "isDependent": True, "fields": []},
        {"name": "Invoices", "recordCount": 100, "isDependent": True, "fields": [
            {"name": "PO_MATCHED", "type": "enum", "distribution": [
                {"label": "PO_MATCHED", "percentage": 90},
                {"label": "PO_NOT_MATCHED", "percentage": 10},
            ]},
        ]},
    ],
    "Contract Mismatch (20%)": [
        {"name": "Vendors", "recordCount": 50, "isDependent": False, "fields": []},
        {"name": "Line Items", "recordCount": 100, "isDependent": False, "fields": []},
        {"name": "POs", "recordCount": 50, "isDependent": True, "fields": []},
        {"name": "Contracts", "recordCount": 20, "isDependent": True, "fields": []},
        {"name": "Invoices", "recordCount": 100, "isDependent": True, "fields": [
            {"name": "PO_MATCHED", "type": "enum", "distribution": [{"label": "PO_MATCHED", "percentage": 100}]},
            {"name": "CONTRACT_MATCHED", "type": "enum", "distribution": [
                {"label": "MATCHED_CONTRACT", "percentage": 80},
                {"label": "NOT_MATCHED_CONTRACT", "percentage": 20},
            ]},
        ]},
    ],
    "Price Variance (10%) + High Risk (20%)": [
        {"name": "Vendors", "recordCount": 100, "isDependent": False, "fields": []},
        {"name": "Line Items", "recordCount": 200, "isDependent": False, "fields": []},
        {"name": "POs", "recordCount": 100, "isDependent": True, "fields": []},
        {"name": "Contracts", "recordCount": 10, "isDependent": True, "fields": []},
        {"name": "Invoices", "recordCount": 100, "isDependent": True, "fields": [
            {"name": "PO_MATCHED", "type": "enum", "distribution": [{"label": "PO_MATCHED", "percentage": 100}]},
            {"name": "HIGH_RISK", "type": "enum", "distribution": [
                {"label": "HIGH_RISK", "percentage": 20},
                {"label": "NORMAL", "percentage": 80},
            ]},
            {"name": "PRICE_VARIENCE", "type": "enum", "distribution": [
                {"label": "PRICE_VARIENCE", "percentage": 10},
                {"label": "NONE", "percentage": 90},
            ]},
        ]},
    ],
}

DEFAULT_JSON = json.dumps(SAMPLE_SCENARIOS["Price Variance (10%) + High Risk (20%)"], indent=2)

# -----------------------
# Helpers
# -----------------------
def list_saved_scenarios():
    return sorted([p.name for p in SCENARIO_DIR.glob("*.json")])

def safe_filename(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", " "):
            keep.append(ch)
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned if cleaned else "scenario"

def get_entity_count(config, entity_name: str, default: int) -> int:
    for e in config:
        if str(e.get("name", "")).strip().lower() == entity_name.lower():
            return int(e.get("recordCount", default))
    return default

def derive_settings_from_json(config):
    vendors = get_entity_count(config, "Vendors", 10)
    contracts = get_entity_count(config, "Contracts", 10)
    pos = get_entity_count(config, "POs", 10)
    line_items = get_entity_count(config, "Line Items", 0)

    lines_per_po = 2
    if pos > 0 and line_items > 0:
        lines_per_po = max(2, int(round(line_items / pos)))

    return {"seed": 42, "vendor_count": vendors, "contract_count": contracts, "po_count": pos, "lines_per_po": lines_per_po}

def simple_explain(kpis):
    invoice_count = int(kpis.get("invoice_count", 0))
    validation_counts = kpis.get("validation_counts", {})
    reason_counts = kpis.get("reason_counts", {})
    scenario_counts = kpis.get("scenario_counts", {})

    passed = int(validation_counts.get("PASS", 0))
    failed = int(validation_counts.get("FAIL", 0))

    no_po = int(reason_counts.get("NO_PO", 0))
    contract = int(reason_counts.get("CONTRACT_MISMATCH", 0))
    price = int(reason_counts.get("PRICE_VARIANCE", 0))
    qty = int(reason_counts.get("QTY_VARIANCE", 0))
    high_risk = sum(v for k, v in scenario_counts.items() if "HIGH_RISK" in str(k))

    return "\n".join([
        "**Summary**",
        f"- Total invoices: **{invoice_count}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{failed}**",
        "",
        "**Why invoices failed**",
        f"- NO_PO: **{no_po}**",
        f"- CONTRACT_MISMATCH: **{contract}**",
        f"- PRICE_VARIANCE: **{price}**",
        f"- QTY_VARIANCE: **{qty}**",
        "",
        "**Risk (not always a failure)**",
        f"- HIGH_RISK: **{high_risk}** (low OCR confidence)",
        "- HIGH_RISK can overlap with failures.",
    ])

# -----------------------
# Session state init
# -----------------------
if "config_text" not in st.session_state:
    st.session_state["config_text"] = DEFAULT_JSON
if "last_zip_path" not in st.session_state:
    st.session_state["last_zip_path"] = None

if "selected_saved" not in st.session_state:
    st.session_state["selected_saved"] = "(none)"
if "last_loaded_saved" not in st.session_state:
    st.session_state["last_loaded_saved"] = "(none)"

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# flags to safely reset saved dropdown BEFORE widgets render
if "pending_reset_saved" not in st.session_state:
    st.session_state["pending_reset_saved"] = False
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

# IMPORTANT: apply pending reset BEFORE widgets are created
if st.session_state["pending_reset_saved"]:
    st.session_state["selected_saved"] = "(none)"
    st.session_state["last_loaded_saved"] = "(none)"
    st.session_state["pending_reset_saved"] = False

# -----------------------
# Clear callback
# -----------------------
def clear_all():
    st.session_state["config_text"] = ""
    st.session_state["pending_reset_saved"] = True
    st.session_state["uploader_key"] += 1
    st.session_state["last_zip_path"] = None
    st.session_state["last_uploaded_name"] = None

# -----------------------
# Layout
# -----------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Scenario")

    # Toolbar row: Sample | Load | Saved | Clear
    c1, c2, c3, c4 = st.columns([3.2, 1.2, 2.0, 1.2])

    with c1:
        selected_sample = st.selectbox("Sample scenario", list(SAMPLE_SCENARIOS.keys()), key="selected_sample")
    with c2:
        if st.button("Load", use_container_width=True):
            st.session_state["config_text"] = json.dumps(SAMPLE_SCENARIOS[selected_sample], indent=2)
            st.session_state["pending_reset_saved"] = True
            st.session_state["uploader_key"] += 1
            st.rerun()
    with c3:
        saved_list = list_saved_scenarios()
        st.selectbox("Saved", ["(none)"] + saved_list, key="selected_saved")
    with c4:
        st.button("Clear", use_container_width=True, on_click=clear_all)

    # If saved scenario changed, load it (no resetting here)
    if st.session_state["selected_saved"] != st.session_state["last_loaded_saved"]:
        st.session_state["last_loaded_saved"] = st.session_state["selected_saved"]
        if st.session_state["selected_saved"] != "(none)":
            st.session_state["config_text"] = (SCENARIO_DIR / st.session_state["selected_saved"]).read_text(encoding="utf-8")

    # Upload below toolbar
    uploaded = st.file_uploader(
        "Upload Lovable JSON",
        type=["json"],
        key=f"uploader_{st.session_state['uploader_key']}"
    )
    # Read file and then trigger a rerun so reset happens BEFORE widgets render next run
    if uploaded is not None and uploaded.name != st.session_state["last_uploaded_name"]:
        st.session_state["last_uploaded_name"] = uploaded.name
        st.session_state["config_text"] = uploaded.read().decode("utf-8")
        st.session_state["pending_reset_saved"] = True
        st.rerun()

    # JSON editor
    st.text_area("Lovable JSON", key="config_text", height=360)

    # Save scenario row
    s1, s2, s3 = st.columns([3.2, 1.4, 1.4])
    with s1:
        scenario_name = st.text_input("Save as", value="", placeholder="e.g., demo_no_po_10pct")
    with s2:
        save_clicked = st.button("Save scenario", use_container_width=True)
    with s3:
        if st.button("Load default", use_container_width=True):
            st.session_state["config_text"] = DEFAULT_JSON
            st.session_state["pending_reset_saved"] = True
            st.session_state["uploader_key"] += 1
            st.rerun()

    if save_clicked:
        if not scenario_name.strip():
            st.error("Enter a scenario name before saving.")
        else:
            fname = safe_filename(scenario_name) + ".json"
            try:
                json.loads(st.session_state["config_text"])  # validate JSON
                (SCENARIO_DIR / fname).write_text(st.session_state["config_text"], encoding="utf-8")
                st.success(f"Saved: scenarios/{fname}")
            except Exception as e:
                st.error(f"Cannot save (invalid JSON): {e}")

    # Parse JSON
    if not st.session_state["config_text"].strip():
        st.info("Paste / upload JSON or load a sample scenario.")
        st.stop()

    try:
        config = json.loads(st.session_state["config_text"])
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    settings = derive_settings_from_json(config)
    st.caption(
        f"Auto settings → Seed {settings['seed']} | Vendors {settings['vendor_count']} | "
        f"Contracts {settings['contract_count']} | POs {settings['po_count']} | Lines/PO {settings['lines_per_po']}"
    )

    with st.expander("Advanced overrides", expanded=False):
        a1, a2, a3, a4, a5 = st.columns(5)
        settings["seed"] = int(a1.number_input("Seed", value=int(settings["seed"]), step=1))
        settings["vendor_count"] = int(a2.number_input("Vendors", value=int(settings["vendor_count"]), step=1, min_value=1))
        settings["contract_count"] = int(a3.number_input("Contracts", value=int(settings["contract_count"]), step=1, min_value=1))
        settings["po_count"] = int(a4.number_input("POs", value=int(settings["po_count"]), step=1, min_value=1))
        settings["lines_per_po"] = int(a5.number_input("Lines/PO", value=int(settings["lines_per_po"]), step=1, min_value=1))

with right:
    st.subheader("Output")

    generate = st.button("Generate CSV + ZIP", type="primary", use_container_width=True)

    if generate:
        vendors, contracts, pos, po_lines, invoices, invoice_lines, validations, kpis = generate_from_config(
            config=config,
            seed=settings["seed"],
            vendor_count=settings["vendor_count"],
            contract_count=settings["contract_count"],
            po_count=settings["po_count"],
            lines_per_po=settings["lines_per_po"],
        )

        pass_n = int(kpis.get("validation_counts", {}).get("PASS", 0))
        fail_n = int(kpis.get("validation_counts", {}).get("FAIL", 0))
        high_risk_n = sum(v for k, v in kpis.get("scenario_counts", {}).items() if "HIGH_RISK" in str(k))
        inv_n = int(kpis.get("invoice_count", 0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invoices", inv_n)
        m2.metric("PASS", pass_n)
        m3.metric("FAIL", fail_n)
        m4.metric("HIGH_RISK", high_risk_n)

        st.markdown("### Failure reasons")
        reasons = kpis.get("reason_counts", {})
        reason_rows = [{"reason": k, "count": int(reasons.get(k, 0))}
                       for k in ["NO_PO", "CONTRACT_MISMATCH", "PRICE_VARIANCE", "QTY_VARIANCE"]]
        st.dataframe(reason_rows, use_container_width=True, hide_index=True)

        st.markdown("### Risk summary")
        st.write({"HIGH_RISK_count": high_risk_n, "avg_ocr_confidence": kpis.get("avg_ocr_confidence", None)})

        st.markdown("### Simple explanation")
        st.markdown(simple_explain(kpis))

        outdir = Path("outputs")
        zip_path = save_and_zip(outdir, vendors, contracts, pos, po_lines, invoices, invoice_lines, validations, kpis)
        st.session_state["last_zip_path"] = str(zip_path)
        st.success("Generated successfully.")

    if st.session_state["last_zip_path"]:
        with open(st.session_state["last_zip_path"], "rb") as f:
            st.download_button(
                "Download ap_demo_data.zip",
                data=f,
                file_name="ap_demo_data.zip",
                mime="application/zip",
                use_container_width=True
            )

    with st.expander("Advanced details (tables)", expanded=False):
        if "kpis" in locals():
            st.subheader("KPIs (raw)")
            st.json(kpis)
            st.subheader("Invoices (first 50)")
            st.dataframe(invoices.head(50), use_container_width=True)
            st.subheader("Validations (first 50)")
            st.dataframe(validations.head(50), use_container_width=True)
            st.subheader("Invoice lines (first 50)")
            st.dataframe(invoice_lines.head(50), use_container_width=True)
        else:
            st.info("Generate once to see tables here.")