import json
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# -------------------------
# Normalization helpers
# -------------------------
def norm(s: str) -> str:
    return str(s).strip().upper().replace("-", "_").replace(" ", "_")


def largest_remainder(distribution: List[Dict[str, Any]], total: int) -> Dict[str, int]:
    raw = [(norm(d["label"]), float(d["percentage"]) * total / 100.0) for d in distribution]
    floors = {label: int(val) for label, val in raw}
    remainder = total - sum(floors.values())
    fracs = sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True)
    for i in range(remainder):
        floors[fracs[i][0]] += 1
    return floors


def allocate_ids(n: int, distribution: List[Dict[str, Any]], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Returns mapping: label -> array(invoice_indexes) using largest remainder + shuffled ids.
    """
    counts = largest_remainder(distribution, n)
    ids = np.arange(n)
    rng.shuffle(ids)

    out = {}
    start = 0
    for label, count in counts.items():
        out[label] = ids[start:start + count]
        start += count
    return out


# -------------------------
# Invoice profile (orthogonal fields)
# -------------------------
@dataclass
class InvoiceProfile:
    po_matched: bool = True
    contract_matched: bool = True
    price_variance: bool = False
    qty_variance: bool = False
    high_risk: bool = False

def build_profiles(invoice_entity: Dict[str, Any], seed: int) -> List[InvoiceProfile]:
    """
    DEMO MODE (Strict Quotas, Disjoint Buckets)

    Each field distribution is treated as a quota of invoices.
    Quotas are assigned disjointly in this priority order:

      1) NO_PO (PO_NOT_MATCHED)
      2) CONTRACT_MISMATCH
      3) PRICE_VARIANCE
      4) QTY_VARIANCE
      5) HIGH_RISK (can be overlay OR disjoint; here we keep it overlay on remaining)

    This ensures counts match Lovable sliders exactly, without being reduced by dependencies.
    """
    n = int(invoice_entity["recordCount"])
    rng = np.random.default_rng(seed)

    profiles = [InvoiceProfile() for _ in range(n)]

    # Start: everyone is PO matched, contract matched, no variance, not high risk
    for p in profiles:
        p.po_matched = True
        p.contract_matched = True
        p.price_variance = False
        p.qty_variance = False
        p.high_risk = False

    # Helper to get field distribution by matching name loosely
    def find_field(predicate):
        for f in invoice_entity.get("fields", []):
            name = norm(f.get("name", ""))
            if predicate(name):
                return f
        return None

    # Extract distributions (if present)
    po_field = find_field(lambda name: "PO" in name and "PRICE" not in name and "QTY" not in name)
    contract_field = find_field(lambda name: "CONTRACT" in name)
    price_field = find_field(lambda name: "PRICE" in name)
    qty_field = find_field(lambda name: ("QTY" in name) or ("QUANTITY" in name))
    risk_field = find_field(lambda name: "RISK" in name)

    # Pool of unassigned invoice indexes
    all_ids = np.arange(n)
    rng.shuffle(all_ids)
    available = list(all_ids)  # list for deterministic pop

    def take(k: int) -> List[int]:
        """Take k unique invoice indexes from the available pool."""
        k = max(0, min(k, len(available)))
        chosen = available[:k]
        del available[:k]
        return [int(x) for x in chosen]

    # -----------
    # 1) NO_PO quota
    # -----------
    no_po_count = 0
    if po_field and po_field.get("distribution"):
        counts = largest_remainder(po_field["distribution"], n)
        # interpret NOT_MATCHED as NO_PO
        no_po_count = sum(v for lbl, v in counts.items() if "NOT_MATCHED" in lbl or "NO_PO" in lbl)

    no_po_ids = set(take(no_po_count))
    for i in no_po_ids:
        profiles[i].po_matched = False
        profiles[i].contract_matched = True
        profiles[i].price_variance = False
        profiles[i].qty_variance = False

    # -----------
    # 2) CONTRACT_MISMATCH quota (from remaining)
    # -----------
    contract_mismatch_count = 0
    if contract_field and contract_field.get("distribution"):
        counts = largest_remainder(contract_field["distribution"], n)
        contract_mismatch_count = sum(v for lbl, v in counts.items() if "NOT_MATCHED" in lbl)

    contract_mismatch_ids = set(take(contract_mismatch_count))
    for i in contract_mismatch_ids:
        profiles[i].contract_matched = False

    # -----------
    # 3) PRICE_VARIANCE quota (from remaining)
    # -----------
    price_var_count = 0
    if price_field and price_field.get("distribution"):
        counts = largest_remainder(price_field["distribution"], n)
        # any label containing PRICE but not NONE
        price_var_count = sum(v for lbl, v in counts.items() if "PRICE" in lbl and "NONE" not in lbl)

    price_var_ids = set(take(price_var_count))
    for i in price_var_ids:
        profiles[i].price_variance = True

    # -----------
    # 4) QTY_VARIANCE quota (from remaining)
    # -----------
    qty_var_count = 0
    if qty_field and qty_field.get("distribution"):
        counts = largest_remainder(qty_field["distribution"], n)
        qty_var_count = sum(v for lbl, v in counts.items() if ("QTY" in lbl or "QUANTITY" in lbl) and "NONE" not in lbl)

    qty_var_ids = set(take(qty_var_count))
    for i in qty_var_ids:
        profiles[i].qty_variance = True

    # -----------
    # 5) HIGH_RISK (overlay; does NOT need to be disjoint)
    # -----------
    # For demos, risk is often independent. We'll set high_risk on top of existing profiles.
    high_risk_count = 0
    if risk_field and risk_field.get("distribution"):
        counts = largest_remainder(risk_field["distribution"], n)
        high_risk_count = sum(v for lbl, v in counts.items() if "HIGH" in lbl)

    # choose high risk IDs from ALL invoices (including already assigned) but disjoint selection is optional
    # We'll select from all invoices for exact count.
    if high_risk_count > 0:
        ids = np.arange(n)
        rng.shuffle(ids)
        for i in ids[:high_risk_count]:
            profiles[int(i)].high_risk = True

    return profiles
# -------------------------
# Master data generation
# -------------------------
def generate_master(seed: int, vendor_count: int, contract_count: int, po_count: int, lines_per_po: int):
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    vendors = pd.DataFrame({
        "vendor_id": [f"V{i+1:03}" for i in range(vendor_count)],
        "vendor_name": [fake.company() for _ in range(vendor_count)]
    })

    contracts = pd.DataFrame({
        "contract_id": [f"C{i+1:03}" for i in range(contract_count)],
        "vendor_id": rng.choice(vendors["vendor_id"].values, contract_count)
    })

    pos = pd.DataFrame({
        "po_id": [f"PO{i+1:04}" for i in range(po_count)],
        "vendor_id": rng.choice(vendors["vendor_id"].values, po_count),
        "contract_id": rng.choice(contracts["contract_id"].values, po_count)
    })

    po_lines = []
    for _, po in pos.iterrows():
        for ln in range(1, lines_per_po + 1):
            qty = int(rng.integers(1, 10))
            price = round(float(rng.integers(50, 200)), 2)
            po_lines.append({
                "po_line_id": f"POL{len(po_lines)+1:05}",
                "po_id": po["po_id"],
                "line_number": ln,
                "quantity": qty,
                "unit_price": price
            })
    po_lines = pd.DataFrame(po_lines)

    return vendors, contracts, pos, po_lines


# -------------------------
# Invoice generation + mutations
# -------------------------
def generate_from_config(
    config: List[Dict[str, Any]],
    seed: int = 42,
    vendor_count: int = 10,
    contract_count: int = 10,
    po_count: int = 10,
    lines_per_po: int = 2,
    price_variance_pct_range: Tuple[float, float] = (0.10, 0.30),
    qty_variance_add_range: Tuple[int, int] = (1, 3),
):
    invoice_entity = next(e for e in config if norm(e["name"]) == "INVOICES")
    n = int(invoice_entity["recordCount"])

    profiles = build_profiles(invoice_entity, seed)

    vendors, contracts, pos, po_lines = generate_master(seed, vendor_count, contract_count, po_count, lines_per_po)

    rng = np.random.default_rng(seed + 999)

    invoices = []
    invoice_lines = []

    # quick maps
    po_map = pos.set_index("po_id").to_dict("index")
    po_lines_by_po = {k: v for k, v in po_lines.groupby("po_id")}

    all_contract_ids = contracts["contract_id"].tolist()

    for i in range(n):
        p = profiles[i]
        invoice_id = f"INV{i+1:04}"

        # base vendor
        vendor_id = rng.choice(vendors["vendor_id"].values)

        if p.po_matched:
            po = pos.sample(1, random_state=int(seed + i)).iloc[0]
            po_id = po["po_id"]
            vendor_id = po["vendor_id"]
            po_contract = po["contract_id"]

            # contract
            if p.contract_matched:
                contract_id = po_contract
            else:
                # choose a different contract id
                candidates = [c for c in all_contract_ids if c != po_contract]
                contract_id = rng.choice(candidates) if candidates else po_contract

            # OCR
            ocr_conf = round(float(rng.uniform(0.85, 0.99)), 2)
            if p.high_risk:
                ocr_conf = round(float(rng.uniform(0.30, 0.60)), 2)

            invoices.append({
                "invoice_id": invoice_id,
                "vendor_id": vendor_id,
                "po_id": po_id,
                "contract_id": contract_id,
                "ocr_confidence": ocr_conf,
                # store profile as readable tags
                "scenario": scenario_tag(p)
            })

            # lines copied from PO lines then mutated
            polines = po_lines_by_po[po_id]
            for _, pol in polines.iterrows():
                qty = int(pol["quantity"])
                unit_price = float(pol["unit_price"])

                if p.price_variance:
                    bump = float(rng.uniform(price_variance_pct_range[0], price_variance_pct_range[1]))
                    unit_price = round(unit_price * (1.0 + bump), 2)

                if p.qty_variance:
                    add = int(rng.integers(qty_variance_add_range[0], qty_variance_add_range[1] + 1))
                    qty = qty + add

                invoice_lines.append({
                    "invoice_line_id": f"INVL{len(invoice_lines)+1:05}",
                    "invoice_id": invoice_id,
                    "po_line_id": pol["po_line_id"],
                    "line_number": int(pol["line_number"]),
                    "quantity": qty,
                    "unit_price": unit_price
                })

        else:
            # NO PO invoice: no po_id, no contract_id, no po_line_id
            po_id = None
            contract_id = None
            ocr_conf = round(float(rng.uniform(0.85, 0.99)), 2)
            if p.high_risk:
                ocr_conf = round(float(rng.uniform(0.30, 0.60)), 2)

            invoices.append({
                "invoice_id": invoice_id,
                "vendor_id": vendor_id,
                "po_id": None,
                "contract_id": None,
                "ocr_confidence": ocr_conf,
                "scenario": scenario_tag(p)
            })

            for ln in range(1, lines_per_po + 1):
                qty = int(rng.integers(1, 10))
                unit_price = round(float(rng.uniform(50, 200)), 2)
                invoice_lines.append({
                    "invoice_line_id": f"INVL{len(invoice_lines)+1:05}",
                    "invoice_id": invoice_id,
                    "po_line_id": None,
                    "line_number": ln,
                    "quantity": qty,
                    "unit_price": unit_price
                })

    invoices = pd.DataFrame(invoices)
    invoice_lines = pd.DataFrame(invoice_lines)

    validations = validate(invoices, invoice_lines, pos, po_lines)
    kpis = compute_kpis(invoices, validations)

    return vendors, contracts, pos, po_lines, invoices, invoice_lines, validations, kpis


def scenario_tag(p: InvoiceProfile) -> str:
    tags = []
    tags.append("PO_MATCHED" if p.po_matched else "NO_PO")
    if p.po_matched:
        tags.append("CONTRACT_OK" if p.contract_matched else "CONTRACT_MISMATCH")
        if p.price_variance:
            tags.append("PRICE_VARIANCE")
        if p.qty_variance:
            tags.append("QTY_VARIANCE")
    if p.high_risk:
        tags.append("HIGH_RISK")
    return "|".join(tags)


# -------------------------
# Validation engine (truth based)
# Precedence: NO_PO > CONTRACT_MISMATCH > PRICE_VARIANCE > QTY_VARIANCE > PASS
# -------------------------
def validate(invoices: pd.DataFrame, invoice_lines: pd.DataFrame, pos: pd.DataFrame, po_lines: pd.DataFrame) -> pd.DataFrame:
    po_map = pos.set_index("po_id").to_dict("index")
    po_line_map = po_lines.set_index("po_line_id").to_dict("index")

    rows = []
    for _, inv in invoices.iterrows():
        inv_id = inv["invoice_id"]
        po_id = inv["po_id"]

        # NO_PO
        if pd.isna(po_id) or po_id is None:
            rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "NO_PO"})
            continue

        po = po_map.get(po_id)
        if not po:
            rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "INVALID_PO"})
            continue

        # Contract mismatch check
        if inv["contract_id"] != po["contract_id"]:
            rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "CONTRACT_MISMATCH"})
            continue

        # Price/Qty checks based on linked po_line_id
        inv_lines = invoice_lines[invoice_lines["invoice_id"] == inv_id]

        for _, line in inv_lines.iterrows():
            pol_id = line["po_line_id"]
            if pd.isna(pol_id) or pol_id is None:
                rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "MISSING_PO_LINE"})
                break

            pol = po_line_map.get(pol_id)
            if not pol:
                rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "INVALID_PO_LINE"})
                break

            if float(line["unit_price"]) != float(pol["unit_price"]):
                rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "PRICE_VARIANCE"})
                break

            if int(line["quantity"]) != int(pol["quantity"]):
                rows.append({"invoice_id": inv_id, "status": "FAIL", "reason": "QTY_VARIANCE"})
                break
        else:
            rows.append({"invoice_id": inv_id, "status": "PASS", "reason": "PASS"})

    return pd.DataFrame(rows)


def compute_kpis(invoices: pd.DataFrame, validations: pd.DataFrame) -> Dict[str, Any]:
    return {
        "invoice_count": int(len(invoices)),
        "scenario_counts": invoices["scenario"].value_counts().to_dict(),
        "validation_counts": validations["status"].value_counts().to_dict(),
        "reason_counts": validations["reason"].value_counts().to_dict(),
        "avg_ocr_confidence": float(invoices["ocr_confidence"].mean()) if len(invoices) else 0.0,
    }


# -------------------------
# Output helpers
# -------------------------
def save_and_zip(
    outdir: Path,
    vendors: pd.DataFrame,
    contracts: pd.DataFrame,
    pos: pd.DataFrame,
    po_lines: pd.DataFrame,
    invoices: pd.DataFrame,
    invoice_lines: pd.DataFrame,
    validations: pd.DataFrame,
    kpis: Dict[str, Any]
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    vendors.to_csv(outdir / "vendors.csv", index=False)
    contracts.to_csv(outdir / "contracts.csv", index=False)
    pos.to_csv(outdir / "pos.csv", index=False)
    po_lines.to_csv(outdir / "po_lines.csv", index=False)
    invoices.to_csv(outdir / "invoices.csv", index=False)
    invoice_lines.to_csv(outdir / "invoice_lines.csv", index=False)
    validations.to_csv(outdir / "invoice_validations.csv", index=False)

    with open(outdir / "kpis.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)

    zip_path = outdir / "ap_demo_data.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in [
            "vendors.csv", "contracts.csv", "pos.csv", "po_lines.csv",
            "invoices.csv", "invoice_lines.csv", "invoice_validations.csv", "kpis.json"
        ]:
            z.write(outdir / fname, arcname=fname)

    return zip_path