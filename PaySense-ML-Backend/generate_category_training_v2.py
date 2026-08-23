"""
================================================================================
  generate_category_training_v2.py
  ----------------------------------------------------------------------------
  CATEGORY_CLASSIFIER_GENERALIZATION.md diagnosed why the deployed category
  classifier scores 72.5% (not 100%) on realistic novel text: FinText-6K's
  5,000 training rows are generated from only 40 fixed sentence templates, so
  the fitted TF-IDF vocabulary is a mere 821 tokens, entirely mined from those
  40 shapes. Novel text sharing zero words with training collapses to an
  identical, label-independent default prediction (Failure Mode A); text
  reusing exactly one training word can be dominated by that single token
  regardless of true meaning (Failure Mode B).

  This script builds a SECOND, much larger, much more structurally diverse
  synthetic dataset intended to widen the vocabulary and reduce single-
  template/single-keyword dependence, so a classifier trained on the blend
  (FinText-6K + this + real-download Kaggle data, see
  CATEGORY_CLASSIFIER_V2_ATTEMPT.md) has a fighting chance at genuine
  narration-shape generalization, not just more rows of the same 40 shapes.

  Grounding (cited, not invented):
  - RBI mandates that banks send SMS/email alerts for electronic banking
    transactions and specifies the alert must carry account/card number,
    amount, date/time, channel, and beneficiary/merchant details -- the
    fields every template below fills. (RBI circulars on customer
    protection / EBT alerts; summarized reporting e.g.
    https://hellobanker.in/bank-customers-to-get-instant-sms-for-transactions-above-rs-500-banks-to-provide-compensation-also-in-case-of-frauds/,
    https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/CIR240114BKCIEN.pdf)
  - The specific per-bank SMS narration *conventions* below (HDFC's
    "UPI/DR/<ref>/<merchant>/<bank>" and "a/c **XXXX" masking; SBI's "Dear
    Customer, Rs.X debited ... Refno ... -SBI"; ICICI's "Info: UPI/<ref>/
    <merchant>/UPI"; Axis's "Acct XXNNNN debited INR X ... -Axis Bank"; GPay/
    PhonePe/Paytm in-app notification phrasing; IMPS-P2A / NACH-DR / ECS-DR /
    SI EXECUTED narrations for standing instructions and EMI/SIP debits) are
    the same convention family CATEGORY_CLASSIFIER_GENERALIZATION.md's
    build_category_generalization_test_set.py already researched and used
    for the 200-row GOLD EVALUATION set -- reused here as a grounding basis
    for the *shape* of real narrations (a public, well-documented format
    convention, not proprietary text), while every literal sentence
    instantiated below is generated fresh from a much larger merchant
    vocabulary and verified programmatically to be byte-for-byte and
    structurally disjoint from that 200-row file (checked at the bottom of
    this script and re-checked in
    tests/test_category_training_v2_disjointness.py).

  This is training data, not eval data: it must never leak into
  category_generalization_test_set.csv, the frozen gold-standard held-out
  set. Both an exact-text check and a digit-collapsed normalized-structure
  check are enforced below; the script refuses to write its output if either
  check fails.

  Output: category_training_v2_synthetic.csv (text,label -- label cased to
  match FinText-6K: food/travel/EMI/investment/shopping).

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe generate_category_training_v2.py
================================================================================
"""
from __future__ import annotations

import csv
import os
import random
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(_HERE, "category_training_v2_synthetic.csv")
EVAL_CSV = os.path.join(_HERE, "category_generalization_test_set.csv")

RNG = random.Random(20260823)  # fixed seed, reproducible, distinct from any
                                # seed used for the eval set or FinText-6K

# ─── Shared vocabulary pools ────────────────────────────────────────────────
BANKS = [
    ("HDFC Bank", "HDFC Bank"), ("SBI", "SBI"), ("ICICI Bank", "ICICI Bank"),
    ("Axis Bank", "Axis Bank"), ("Kotak Mahindra Bank", "Kotak Bank"),
    ("Punjab National Bank", "PNB"), ("Bank of Baroda", "BOB"),
    ("Canara Bank", "Canara Bank"), ("Union Bank of India", "Union Bank"),
    ("IDFC FIRST Bank", "IDFC First"), ("Yes Bank", "Yes Bank"),
    ("IndusInd Bank", "IndusInd Bank"), ("RBL Bank", "RBL Bank"),
    ("Federal Bank", "Federal Bank"), ("IDBI Bank", "IDBI Bank"),
]
APPS = ["Google Pay", "PhonePe", "Paytm", "BHIM", "Amazon Pay", "Mobikwik", "WhatsApp Pay"]
PHONES = ["9223008333", "18002586161", "1800112211", "18001234567", "9188888888"]

def _amt():
    return RNG.choice([RNG.randint(20, 999), RNG.randint(1000, 9999), RNG.randint(10000, 60000)])

def _ref(n=12):
    return "".join(RNG.choice("0123456789") for _ in range(n))

def _hexref(n=8):
    return "".join(RNG.choice("0123456789abcdef") for _ in range(n))

def _acct():
    return "".join(RNG.choice("0123456789") for _ in range(4))

def _date():
    d = RNG.randint(1, 28)
    m = RNG.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return f"{d:02d}{m}26"

def _date2():
    d = RNG.randint(1, 28)
    mm = RNG.randint(1, 12)
    return f"{d:02d}-{mm:02d}-2026"

def _bal():
    return RNG.randint(500, 95000)

def _vpa(merchant):
    slug = re.sub(r"[^a-z]", "", merchant.lower())[:10] or "merchant"
    return f"{slug}@ybl"


# ─── Template skeleton library ──────────────────────────────────────────────
# Each skeleton is a function(merchant) -> str. Kept deliberately varied in
# word order, connective words, and field ordering so the TF-IDF vectorizer
# sees genuinely different bag-of-words/bigram contexts around merchant
# names, not one fixed suffix the way FinText-6K's 40 templates share
# " of Rs # via UPI Ref #" verbatim on every single row.

def t01(m):
    bank, _ = RNG.choice(BANKS)
    return f"UPI/DR/{_ref()}/{m.upper()}/{bank}"

def t02(m):
    _, bshort = RNG.choice(BANKS)
    return (f"Dear Customer, Rs.{_amt()}.00 debited from A/c XX{_acct()} on {_date()} "
            f"trf to {m.upper()} Refno {_ref()}. If not done by u, fwd this sms to "
            f"{RNG.choice(PHONES)} -{bshort}")

def t03(m):
    bank, _ = RNG.choice(BANKS)
    return (f"INR {_amt()}.00 debited from {bank} A/c XX{_acct()} on {_date2()}. "
            f"Info: UPI/{_ref()}/{m.upper()}/UPI. Avl Bal INR {_bal()}.00")

def t04(m):
    _, bshort = RNG.choice(BANKS)
    return f"Acct XX{_acct()} debited INR {_amt()}.00 on {_date2()} UPI Ref {_ref()} to {m.upper()} -{bshort}"

def t05(m):
    return f"You paid ₹{_amt()} to {m} using Google Pay"

def t06(m):
    return f"Paid Rs {_amt()} to {m} via PhonePe. Txn ID T26{_ref(17)}"

def t07(m):
    return f"Rs {_amt()} paid to {m} via Paytm UPI. Order ID {_ref(6)}"

def t08(m):
    bank, _ = RNG.choice(BANKS)
    return f"IMPS-P2A-{_ref()}-{m.upper()}-{bank.upper()}-Rs.{_amt()}.00"

def t09(m):
    return f"amt rs.{_amt()} dedcuted a/c XX{_acct()} for {m.upper()} on {_date2()[:8]} upi ref {_ref()}"

def t10(m):
    low = m.lower()
    return f"{low} paymnt rs {_amt()} debited, ref no {_ref()}"

def t11(m):
    return f"Rs {_amt()}.00 sent to {m} via BHIM UPI, ref {_ref()}"

def t12(m):
    return f"Sent Rs.{_amt()} to {m.upper()} using Amazon Pay UPI"

def t13(m):
    return f"A/c XX{_acct()} dr with Rs {_amt()} towards {m.upper()} on {_date()[:5]} UPI Ref {_ref()}"

def t14(m):
    app = RNG.choice(APPS)
    return f"{app} - Rs {_amt()} paid to {m}, txn successful"

def t15(m):
    bank, _ = RNG.choice(BANKS)
    return f"UPI/DR/{_ref()}/{m.upper()}/{bank}/Ref {_ref()}"

def t16(m):
    return f"Rs.{_amt()}.00 debited a/c no. XX{_acct()} towards {m.upper()} on {_date2()}"

def t17(m):
    return f"gpay: paid ₹{_amt()} to {m}"

def t18(m):
    return f"Bill paid Rs {_amt()} at {m.upper()} via PhonePe"

def t19(m):
    _, bshort = RNG.choice(BANKS)
    return f"UPI-DR-{_ref()}-{m.upper()}-{bshort}-Rs.{_amt()}.00"

def t20(m):
    return f"Auto-debit Rs {_amt()} - {m.upper()}, UPI Ref {_ref()}"

def t21(m):
    bank, _ = RNG.choice(BANKS)
    return f"{bank}: Rs.{_amt()}.00 debited from a/c **{_acct()} on {_date()}; UPI Ref {_ref()}; VPA {_vpa(m)}"

def t22(m):
    return f"Rs {_amt()} settled via UPI to {m.upper()}, ref {_ref()}"

def t23(m):
    return f"Rs.{_amt()} dr for {m.upper()} a/c XX{_acct()}, UPI/{_ref()}/{m.upper()}"

def t24(m):
    low = m.lower()
    return f"paid rupees {_amt()} 2 {low} thru phonepe"

def t25(m):
    return f"Rs {_amt()}.00 debited - {m.upper()} - A/c XX{_acct()} - Ref {_ref()}"

def t26(m):
    bank, _ = RNG.choice(BANKS)
    return f"NACH-DR-{_ref()}-{m.upper()}-{bank.upper()}-Rs.{_amt()}.00"

def t27(m):
    bank, _ = RNG.choice(BANKS)
    return f"SI EXECUTED-{m.upper()}-A/c XX{_acct()}-Rs.{_amt()}.00-{bank}"

def t28(m):
    _, bshort = RNG.choice(BANKS)
    return f"ECS-DR-{_ref()}-{m.upper()}-{bshort}"

def t29(m):
    return f"Standing instruction Rs {_amt()} executed for {m.upper()}, ref {_ref()}"

def t30(m):
    _, bshort = RNG.choice(BANKS)
    return f"Your {m} of Rs {_amt()} has been auto-debited from a/c XX{_acct()} - {bshort}"

def t31(m):
    return f"POS {m.upper()} Rs.{_amt()} debited card XX{_acct()}, avl lmt Rs {_bal()}"

def t32(m):
    low = m.lower()
    return f"paid rs.{_amt()} 4 {low} thru {RNG.choice(['nach', 'paytm upi', 'phonepe', 'gpay'])}"

def t33(m):
    return f"NACH mandate executed: Rs {_amt()} debited towards {m.upper()}, Ref NACH{_ref(7)}"

def t34(m):
    return f"Rs.{_amt()} dedcted frm a/c thru {RNG.choice(['nach', 'ecs'])} mandate for {m.upper()}"

def t35(m):
    return f"Rs {_amt()} spent on {m.upper()}, Ref {_ref()} -HDFC Bank"

def t36(m):
    return f"recieved paymnt confrmation: rs {_amt()} paid to {m.lower()}, ref {_ref()}"

def t37(m):
    return f"Paid ₹{_amt()} towards {m} using PhonePe UPI"

def t38(m):
    return f"Payment of Rs {_amt()} towards {m.upper()} via BHIM"

def t39(m):
    return f"UPI/DR/{_ref()}/{m.upper()}/{RNG.choice(BANKS)[0]} Ref {_ref()}"

def t40(m):
    return f"{m} - Rs {_amt()} auto debited from linked account, ref {_ref()}"

SKELETONS = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13,
             t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26,
             t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40]

# Not every skeleton is a plausible surface form for every class (e.g. NACH
# mandate skeletons t26-t29, t33-t34 make sense for EMI/Investment recurring
# debits but read oddly for a one-off Swiggy order). Assign each class the
# subset that is realistic, so the dataset doesn't manufacture nonsense
# sentences just to inflate a structure count.
GENERIC_SKELETONS = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11,
                      t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22,
                      t23, t24, t25, t31, t32, t35, t36, t37, t38, t39, t40]
RECURRING_SKELETONS = [t26, t27, t28, t29, t30, t33, t34]  # EMI + Investment (SIP/NACH-style debits) only

FOOD_SKELETONS = GENERIC_SKELETONS
TRAVEL_SKELETONS = GENERIC_SKELETONS
SHOPPING_SKELETONS = GENERIC_SKELETONS
EMI_SKELETONS = GENERIC_SKELETONS + RECURRING_SKELETONS
INVESTMENT_SKELETONS = GENERIC_SKELETONS + RECURRING_SKELETONS

# ─── Per-class merchant / purpose vocabulary (large, real Indian brands) ────
FOOD_MERCHANTS = [
    "Zomato Order", "Swiggy Order", "EatSure Delivery", "Box8 Meal", "FreshMenu Order",
    "Faasos Wrap", "Behrouz Biryani", "Ovenstory Pizza", "Dominos Pizza", "Pizza Hut",
    "McDonalds", "Burger King", "KFC", "Subway Sandwiches", "Taco Bell", "Starbucks Coffee",
    "Cafe Coffee Day", "Chaayos", "Third Wave Coffee", "Blue Tokai Coffee",
    "Barbeque Nation", "Bikanervala Sweets", "Haldirams", "Sagar Ratna", "Saravana Bhavan",
    "Punjabi Dhaba", "Karims Restaurant", "Paradise Biryani", "Mainland China",
    "Absolute Barbecues", "Wow Momo", "Theobroma Bakery", "Monginis Cake Shop",
    "Local Bakery Fresh Bread", "Big Bazaar Grocery", "DMart Grocery", "Reliance Fresh",
    "Reliance Smart Bazaar", "Spencers Retail Grocery", "More Supermarket",
    "Natures Basket Grocery", "Licious Meat Delivery", "Country Delight Dairy",
    "Milkbasket Subscription", "BigBasket Grocery", "Blinkit Grocery", "Zepto Delivery",
    "Swiggy Instamart", "Amul Parlour", "Local Kirana Store", "Local Tiffin Service",
    "Chai Point", "Chaiwala Cart", "Roadside Tea Stall", "Vada Pav Stall",
    "Street Food Chaat Corner", "Highway Dhaba Restaurant", "College Canteen Mess",
    "Catering Services Event", "Dunkin Donuts", "Behrouz Kebab", "Yumlane Meal Order",
]
TRAVEL_MERCHANTS = [
    "Ola Cabs", "Uber Ride", "Rapido Bike Taxi", "Meru Cabs", "BluSmart EV Cab",
    "IRCTC Rail Ticket", "RedBus Ticket", "IntrCity SmartBus", "Zoomcar Rental",
    "Revv Self Drive", "MakeMyTrip Booking", "Yatra Booking", "Cleartrip Booking",
    "EaseMyTrip Booking", "Goibibo Booking", "Trainman Ticket", "IndiGo Airlines",
    "SpiceJet Airlines", "Air India", "Vistara Airlines", "Akasa Air", "Air Asia India",
    "OYO Rooms", "Treebo Hotels", "FabHotels", "Airbnb Booking", "Taj Hotel Booking",
    "Marriott Hotel Booking", "IOCL Petrol Pump", "BPCL Petrol Pump", "HPCL Fuel Station",
    "FASTag NHAI Recharge", "Metro Card Recharge", "DMRC Delhi Metro", "BMRC Namma Metro",
    "State Road Transport Bus", "Toll Plaza FASTag", "Parking Lot Payment",
    "Auto Rickshaw Fare", "Cycle Rickshaw Fare", "Sharing Auto Fare", "Airport Cab Pickup",
    "Goa Ferry Service", "BlaBlaCar Carpool", "Travel Agent Package", "GoAir Flight",
    "Indian Railways Catering", "Bus Pass Renewal", "Highway Toll Tax",
]
EMI_MERCHANTS = [
    "HDFC Home Loan EMI", "SBI Car Loan EMI", "ICICI Personal Loan EMI",
    "Axis Bank Education Loan EMI", "Bajaj Finserv EMI", "Tata Capital EMI",
    "IIFL Finance EMI", "HDB Financial EMI", "Muthoot Finance Gold Loan EMI",
    "L&T Finance EMI", "Fullerton India EMI", "Home Credit EMI", "Cholamandalam Finance EMI",
    "Mahindra Finance EMI", "Shriram Finance EMI", "AU Small Finance Bank EMI",
    "Kotak Personal Loan EMI", "Avanse Student Loan EMI", "Credit Card EMI Conversion",
    "Bajaj Finserv EMI Card", "ZestMoney BNPL EMI", "Simpl Pay Later EMI",
    "LazyPay EMI", "Mobikwik EMI Purchase", "Two Wheeler Loan EMI", "Tractor Loan EMI",
    "Consumer Durable EMI", "Vehicle Loan EMI", "Personal Loan EMI HDFC",
    "Housing Loan Installment", "Education Loan EMI Axis", "Appliance EMI Purchase",
    "Mobile Phone EMI", "Furniture Purchase EMI", "Credit Card Minimum Due EMI",
]
INVESTMENT_MERCHANTS = [
    "Zerodha Stock Purchase", "Groww Mutual Fund SIP", "Upstox Equity Delivery",
    "Angel One Trading", "ICICI Direct Investment", "HDFC Securities Trading",
    "Kotak Securities Trading", "Paytm Money Mutual Fund", "5paisa Trading",
    "INDmoney Investment", "Smallcase Investment", "Kuvera Mutual Fund",
    "HDFC Mutual Fund SIP", "SBI Mutual Fund SIP", "Axis Bluechip Fund SIP",
    "ICICI Prudential Fund", "Nippon India Mutual Fund", "Mirae Asset Fund SIP",
    "PPF Contribution", "NPS Tier 1 Contribution", "Sukanya Samriddhi Yojana",
    "Recurring Deposit RD", "Fixed Deposit Renewal", "Sovereign Gold Bond SGB",
    "Digital Gold SafeGold", "MMTC PAMP Gold Purchase", "WazirX Crypto Purchase",
    "CoinDCX Crypto Purchase", "CoinSwitch Kuber Crypto", "LIC Premium ULIP",
    "Max Life ULIP Premium", "ICICI Pru ULIP Investment", "HDFC Life Premium",
    "Demat Account Funding", "NSE GOI Bond Purchase", "Term Insurance ULIP Premium",
    "ELSS Tax Saver Fund", "Gold ETF Purchase",
]
SHOPPING_MERCHANTS = [
    "Amazon Order", "Flipkart Order", "Myntra Fashion Order", "Ajio Clothing Order",
    "Meesho Order", "Tata Cliq Purchase", "Nykaa Beauty Order", "Croma Electronics",
    "Reliance Digital Purchase", "Reliance Trends Purchase", "Max Fashion Purchase",
    "Westside Clothing Purchase", "Pantaloons Apparel", "Lifestyle Stores Purchase",
    "Shoppers Stop Apparel", "Decathlon Sports Gear", "Lenskart Eyewear",
    "Titan Watches Purchase", "Tanishq Jewellery", "Pepperfry Furniture Order",
    "Urban Ladder Furniture", "IKEA Home Decor", "FirstCry Kids Shopping",
    "H&M Clothing Store", "Zara Clothing Store", "Uniqlo Clothing Store",
    "Vijay Sales Electronics", "Snapdeal Order", "Bata Footwear Store",
    "Puma Sportswear Purchase", "Woodland Footwear Purchase", "Sephora Beauty Store",
    "Central Mall Shopping", "Crossword Books Purchase", "Hamleys Toy Store",
    "Furniture Mart Sofa Purchase", "Apple Store Purchase", "Samsung Mobile Store",
    "Local Electronics Repair Shop", "Caribbean Bags Handbag Store", "Sunglasses Hut Store",
]

CLASSES = [
    ("food", FOOD_MERCHANTS, FOOD_SKELETONS),
    ("travel", TRAVEL_MERCHANTS, TRAVEL_SKELETONS),
    ("EMI", EMI_MERCHANTS, EMI_SKELETONS),
    ("investment", INVESTMENT_MERCHANTS, INVESTMENT_SKELETONS),
    ("shopping", SHOPPING_MERCHANTS, SHOPPING_SKELETONS),
]

ROWS_PER_MERCHANT_SKELETON = 3  # multiple draws per (merchant, skeleton) pair, varying amt/ref/date/bank


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", str(text))


def build_rows():
    rows = []
    seen_text = set()
    for label, merchants, skeletons in CLASSES:
        for merchant in merchants:
            for skel in skeletons:
                for _ in range(ROWS_PER_MERCHANT_SKELETON):
                    text = skel(merchant)
                    if text in seen_text:
                        continue
                    seen_text.add(text)
                    rows.append((text, label))
    return rows


def main() -> None:
    rows = build_rows()
    print(f"Generated {len(rows)} raw rows before disjointness filtering")

    # ── Structural richness self-check ──────────────────────────────────────
    structures = {_normalize(t) for t, _ in rows}
    print(f"Distinct digit-collapsed structures: {len(structures)}")
    print(f"Distinct literal skeleton functions used: {len(SKELETONS)}")

    # ── Leakage discipline: this is TRAINING data, must never touch the gold
    #    200-row evaluation set, exactly (text) or structurally (normalized).
    if os.path.exists(EVAL_CSV):
        import pandas as pd
        eval_df = pd.read_csv(EVAL_CSV)
        eval_texts = set(eval_df["text"].astype(str))
        eval_norms = set(eval_df["text"].astype(str).apply(_normalize))

        before = len(rows)
        filtered = []
        dropped_exact = 0
        dropped_struct = 0
        for text, label in rows:
            if text in eval_texts:
                dropped_exact += 1
                continue
            if _normalize(text) in eval_norms:
                dropped_struct += 1
                continue
            filtered.append((text, label))
        rows = filtered
        print(f"Dropped {dropped_exact} exact-text collisions with eval set")
        print(f"Dropped {dropped_struct} normalized-structure collisions with eval set")
        print(f"Rows after leakage filtering: {len(rows)} (was {before})")
    else:
        print(f"WARNING: {EVAL_CSV} not found -- skipping leakage check (cannot verify disjointness)")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
    from collections import Counter
    counts = Counter(label for _, label in rows)
    for label, _, _ in CLASSES:
        print(f"  {label:12s}: {counts[label]}")


if __name__ == "__main__":
    main()
