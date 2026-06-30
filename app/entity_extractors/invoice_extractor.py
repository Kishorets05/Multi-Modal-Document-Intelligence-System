"""Invoice entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. LayoutEngine  (PDF real layout OR text heuristics)
2. spaCy NLP     (ORG for vendor/customer, DATE for invoice date, MONEY for totals)
3. Rule Engine   (label-proximity row parser, same as before)
4. ValidationEngine (amounts, dates, invoice numbers + confidence)
"""
from __future__ import annotations

import re
from typing import Final

from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.layout_engine import LayoutBlock
from app.entity_extractors.validation_engine import (
    make_field,
    pdf_bonus,
    validate_date_spacy,
    validate_money,
    validate_org,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Regex — dates, amounts, currency, invoice numbers (validation only)       #
# ─────────────────────────────────────────────────────────────────────────── #

_DATE_RE: Final = re.compile(
    r"(?:"
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r")",
    re.IGNORECASE,
)

_AMOUNT_RE: Final = re.compile(r"[₹$€£¥]?\s*[\d,]+\.?\d*")
_CURRENCY_RE: Final = re.compile(r"(₹|\$|€|£|¥|USD|INR|EUR|GBP|JPY|CAD|AUD)")
_INV_NUM_RE: Final = re.compile(
    r"(?:^|\s)([A-Z]{0,5}-?\d{3,}[A-Z0-9\-/]*)(?:\s|$)"
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Label matchers (rule engine)                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_LBL_INV_NO: Final = re.compile(
    r"^invoice\s*(?:no\.?|number|#|num\.?)\s*[:\-]?\s*(.*)$", re.IGNORECASE
)
_LBL_INV_DATE: Final = re.compile(
    r"^invoice\s*date\s*[:\-]?\s*(.*)$", re.IGNORECASE
)
_LBL_FROM: Final = re.compile(r"^from\s*:?\s*(.*)$", re.IGNORECASE)
_LBL_TO: Final   = re.compile(r"^to\s*:?\s*(.*)$", re.IGNORECASE)
_LBL_SUBTOTAL: Final = re.compile(
    r"^sub[\s\-]?total\s*[:\-]?\s*(.*)$", re.IGNORECASE
)
_LBL_TAX: Final = re.compile(
    r"^(?:tax|gst|vat|cgst|sgst|igst)\s*[:\-]?\s*(.*)$", re.IGNORECASE
)
_LBL_TOTAL: Final = re.compile(
    r"^(?:total\s+due|grand\s+total|total\s+amount|amount\s+due|"
    r"net\s+total|total)\s*[:\-]?\s*(.*)$",
    re.IGNORECASE,
)
_BLOCK_TERM_RE: Final = re.compile(
    r"^(?:invoice\s*(?:number|no\.?|date|#)|order\s*(?:number|no\.?)"
    r"|due\s+date|total\s+due|sub[\s\-]?total|tax|gst|vat"
    r"|grand\s+total|amount\s+due|to\s*:|from\s*:)\b",
    re.IGNORECASE,
)
_NOISE_RE: Final = re.compile(
    r"(?:payment\s+is\s+due|late\s+payment|thanks?\s+for|page\s+\d"
    r"|subject\s+to\s+fees?|per\s+month|per\s+annum"
    r"|bank\s+details?|acc\s*#|bsb\s*#|swift|iban|paid\b"
    r"|www\.|http)",
    re.IGNORECASE,
)
_TABLE_COL_RE: Final = re.compile(
    r"^(?:hrs/qty|qty|quantity|description|service|rate/price|rate|"
    r"price|adjust|item|unit\s*price|amount)\s*$",
    re.IGNORECASE,
)


def _clean_amount(raw: str) -> str:
    m = _AMOUNT_RE.search(raw)
    if not m:
        return ""
    return re.sub(r"([₹$€£¥])\s+", r"\1", m.group().strip())


def _validate_inv_number(value: str) -> str:
    s = value.strip()
    m = _INV_NUM_RE.search(s)
    if m:
        return m.group(1)
    if re.match(r"^[A-Z0-9\-/]+$", s, re.IGNORECASE) and 2 <= len(s) <= 30:
        return s
    return ""


def _next_nonempty(blocks: list[LayoutBlock], after_idx: int) -> str:
    for b in blocks:
        if b.block_index > after_idx and b.stripped:
            return b.stripped
    return ""


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class InvoiceEntityExtractor(BaseEntityExtractor):
    """Extract structured entities from an invoice.

    Priority chain per field
    ------------------------
    vendor/customer : spaCy ORG → label-proximity rule
    invoice_date    : spaCy DATE → regex date
    total_amount    : spaCy MONEY → regex amount
    subtotal/tax    : regex amount (no good NER label)
    invoice_number  : regex + validate_inv_number
    currency        : MONEY entity scan → regex symbol
    """

    _HEADING_NAMES: set[str] = set()

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)
        nlp = self._nlp()

        # ── spaCy pass ────────────────────────────────────────────────────
        doc = self._doc(text)
        spacy_orgs   = self._ents(doc, "ORG")
        spacy_dates  = self._ents(doc, "DATE")
        spacy_money  = self._ents(doc, "MONEY")

        # ── Layout engine ─────────────────────────────────────────────────
        parser = self._build_parser(text)
        blocks = parser.all_blocks

        # ── Initialise outputs ────────────────────────────────────────────
        inv_number   = ""
        inv_date_raw = ""
        vendor_raw   = ""
        customer_raw = ""
        subtotal_raw = ""
        tax_raw      = ""
        total_raw    = ""
        currency_raw = ""

        # ── Currency (regex scan, whole doc) ─────────────────────────────
        # Also check MONEY entities for embedded currency symbols.
        cur_m = _CURRENCY_RE.search(text)
        if cur_m:
            currency_raw = cur_m.group(1)

        # ── Row-by-row label pass (rule engine) ───────────────────────────
        in_from = False
        in_to   = False

        for block in blocks:
            s = block.stripped

            if _NOISE_RE.search(s) or _TABLE_COL_RE.match(s):
                in_from = in_to = False
                continue

            def _nxt() -> str:
                return _next_nonempty(blocks, block.block_index)

            # From (vendor)
            fm = _LBL_FROM.match(s)
            if fm:
                inline = fm.group(1).strip()
                if inline and not _NOISE_RE.search(inline):
                    vendor_raw = inline
                    in_from = False
                else:
                    in_from = True
                in_to = False
                continue
            if in_from:
                if _BLOCK_TERM_RE.match(s):
                    in_from = False
                elif not vendor_raw:
                    vendor_raw = s
                continue

            # To (customer)
            tm = _LBL_TO.match(s)
            if tm:
                inline = tm.group(1).strip()
                if inline and not _NOISE_RE.search(inline):
                    customer_raw = inline
                    in_to = False
                else:
                    in_to = True
                in_from = False
                continue
            if in_to:
                if _BLOCK_TERM_RE.match(s):
                    in_to = False
                elif not customer_raw:
                    customer_raw = s
                continue

            # Invoice Number
            if not inv_number:
                m = _LBL_INV_NO.match(s)
                if m:
                    val = m.group(1).strip() or _nxt()
                    inv_number = _validate_inv_number(val)
                    continue

            # Invoice Date
            if not inv_date_raw:
                m = _LBL_INV_DATE.match(s)
                if m:
                    val = m.group(1).strip() or _nxt()
                    d = _DATE_RE.search(val)
                    inv_date_raw = d.group().strip() if d else val.strip()
                    continue

            # Subtotal
            if not subtotal_raw:
                m = _LBL_SUBTOTAL.match(s)
                if m:
                    val = m.group(1).strip() or _nxt()
                    subtotal_raw = _clean_amount(val)
                    continue

            # Tax
            if not tax_raw:
                m = _LBL_TAX.match(s)
                if m:
                    val = m.group(1).strip() or _nxt()
                    tax_raw = _clean_amount(val)
                    continue

            # Total
            if not total_raw:
                if not re.match(r"^sub", s, re.IGNORECASE):
                    m = _LBL_TOTAL.match(s)
                    if m:
                        val = m.group(1).strip() or _nxt()
                        total_raw = _clean_amount(val)
                        continue

        # ── spaCy enrichment: prefer NER over rule-based when available ───

        # Vendor: spaCy ORG on raw candidate (higher confidence)
        vendor_conf = 0.65 + layout_bonus
        vendor_val  = vendor_raw
        if spacy_orgs and not vendor_raw:
            vendor_val  = spacy_orgs[0]
            vendor_conf = 0.90 + layout_bonus
        elif vendor_raw:
            v_val, v_c = validate_org(vendor_raw, spacy_orgs)
            if v_val:
                vendor_val  = v_val
                vendor_conf = v_c + layout_bonus

        # Customer: same pattern
        customer_conf = 0.65 + layout_bonus
        customer_val  = customer_raw
        if spacy_orgs and not customer_raw:
            # second ORG entity if available
            customer_val  = spacy_orgs[1] if len(spacy_orgs) > 1 else ""
            customer_conf = 0.88 + layout_bonus if customer_val else 0.0
        elif customer_raw:
            c_val, c_c = validate_org(customer_raw, spacy_orgs)
            if c_val:
                customer_val  = c_val
                customer_conf = c_c + layout_bonus

        # Invoice date: spaCy DATE preferred
        date_val  = inv_date_raw
        date_conf = 0.75 + layout_bonus
        if spacy_dates and not inv_date_raw:
            date_val  = spacy_dates[0]
            date_conf = 0.88 + layout_bonus
        elif inv_date_raw:
            dv, dc = validate_date_spacy(inv_date_raw, nlp)
            if dv:
                date_val  = dv
                date_conf = dc + layout_bonus

        # Total: spaCy MONEY preferred
        total_val  = total_raw
        total_conf = 0.75 + layout_bonus
        if spacy_money and not total_raw:
            total_val  = spacy_money[0]
            total_conf = 0.90 + layout_bonus
        elif total_raw:
            tv, tc = validate_money(total_raw)
            if tv:
                total_val  = tv
                total_conf = tc + layout_bonus

        # Subtotal / tax — regex only
        sub_val,  sub_conf  = validate_money(subtotal_raw)
        sub_conf  = round(sub_conf + layout_bonus, 3)
        tax_val,  tax_conf  = validate_money(tax_raw)
        tax_conf  = round(tax_conf + layout_bonus, 3)

        # Invoice number — regex validated
        inv_conf = 0.82 + layout_bonus if inv_number else 0.0

        # Currency
        cur_conf = 0.88 + layout_bonus if currency_raw else 0.0

        return {
            "invoice_number": make_field(inv_number,         round(min(inv_conf, 1.0), 3)),
            "invoice_date":   make_field(date_val,           round(min(date_conf, 1.0), 3)),
            "vendor":         make_field(vendor_val,         round(min(vendor_conf, 1.0), 3)),
            "customer":       make_field(customer_val,       round(min(customer_conf, 1.0), 3)),
            "subtotal":       make_field(sub_val,            sub_conf),
            "tax":            make_field(tax_val,            tax_conf),
            "total_amount":   make_field(total_val,          round(min(total_conf, 1.0), 3)),
            "currency":       make_field(currency_raw,       round(min(cur_conf, 1.0), 3)),
        }
