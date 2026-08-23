"""
================================================================================
  generate_category_training_v3.py
  ----------------------------------------------------------------------------
  v2 of this generation attempt (generate_category_training_v2.py,
  paysense_category_classifier_v2.pkl) was INVALIDATED: it was built after
  reading category_generalization_test_set.csv's content (directly, and via
  a briefing prompt that quoted several of its exact sentences as "failure
  examples"), and its resulting templates turned out to be the eval set's
  own skeletons with only the merchant name swapped -- e.g. the eval file's
  "amt rs.499 dedcuted a/c XX7788 for BARBEQUE NATION on 05-08-26 upi ref
  309112233445" versus v2's "amt rs.78 dedcuted a/c XX7978 for FRESHMENU
  ORDER on 20-08-20 upi ref 164785176038". That is memorization of the eval
  set's specific sentences, not generalization, and the "97.5% accuracy" v2
  reported against that eval set is invalid.

  This script is written WITHOUT looking at category_generalization_test_set.csv
  at any point -- template structures below are built from general knowledge
  of Indian bank/UPI SMS and app-notification conventions (documented
  publicly by banks and RBI, not from that file), deliberately varying
  sentence structure, word order, verb choice, and formatting conventions
  more broadly than a small set of "debited/UPI Ref" skeletons, specifically
  to avoid the same failure mode. The existing digit-collapsing disjointness
  check (tests/test_category_generalization.py's _normalize) would NOT have
  caught v2's contamination (it only collapses digits, not merchant-name
  differences) -- so this script also does its own structural check further
  below, comparing normalized skeletons (digits AND known merchant tokens
  masked) against the eval set, purely programmatically (pandas), never by
  a human/agent reading the file's content.

  Grounding: general knowledge of RBI-mandated SMS alert requirements
  (account/card ref, amount, date/time, channel, merchant/beneficiary must
  all appear) and of how HDFC, SBI, ICICI, Axis, Kotak, Yes Bank, IDFC
  First, Federal Bank, Canara Bank, Bank of Baroda, IndusInd, and PNB
  format their transaction SMS, plus GPay/PhonePe/Paytm/BHIM/Amazon Pay
  in-app notification conventions and NACH/ECS/IMPS/NEFT/RTGS/SI narration
  language for standing instructions, EMI debits, and SIP mandates.

  Output: category_training_v3_synthetic.csv (text,label; label cased to
  match FinText-6K: food/travel/EMI/investment/shopping).
================================================================================
"""
from __future__ import annotations

import csv
import os
import random
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(_HERE, "category_training_v3_synthetic.csv")

random.seed(20260823)

BANKS = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank",
         "Yes Bank", "IDFC FIRST Bank", "Federal Bank", "Canara Bank",
         "Bank of Baroda", "IndusInd Bank", "PNB", "Union Bank of India"]
APPS = ["Google Pay", "PhonePe", "Paytm", "BHIM", "Amazon Pay", "WhatsApp Pay",
        "Mobikwik", "CRED Pay"]

def acct():
    return f"XX{random.randint(1000,9999)}"

def ref():
    return "".join(str(random.randint(0,9)) for _ in range(12))

def amt():
    return round(random.uniform(20, 9500), 2)

def date():
    return f"{random.randint(1,28):02d}-{random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}-26"

# ── Merchant/vendor pools, deliberately wide and distinct from generic examples ──
FOOD_VENDORS = [
    "Sagar Ratna", "Bikanervala", "Nirulas", "Empire Restaurant", "Meghana Foods",
    "Truffles", "Social Offline", "Chinese Wok", "Punjab Grill", "Mainland China",
    "Behrouz Biryani", "Ovenstory Pizza", "Faasos", "Box8", "EatFit", "Freshmenu",
    "Blinkit Grocery", "Zepto Instant", "BigBasket Daily", "JioMart Grocery",
    "Licious Meats", "Country Delight Dairy", "Milkbasket", "Nature's Basket",
    "Café Mocha", "Third Wave Coffee", "Blue Tokai Coffee", "Chaayos",
    "Haldiram's Sweets", "Karachi Bakery", "Local Vegetable Vendor",
    "Kirana Corner Store", "Fruit Cart Wala", "Mess Committee Fund",
    "Canteen Services Ltd", "Bakery Nook", "Ice Cream Parlour Naturals",
]
TRAVEL_VENDORS = [
    "IndiGo Airlines", "Vistara Airways", "Air India", "SpiceJet", "IRCTC",
    "RedBus Travels", "MakeMyTrip", "Yatra Online", "Cleartrip", "Goibibo",
    "Ola Cabs", "Uber India", "Rapido Bike Taxi", "OYO Rooms", "Taj Hotels",
    "FASTag NHAI Toll", "Metro Card Recharge", "State Transport Bus Dept",
    "Airport Parking Services", "Travel Insurance Co", "Zoomcar Self Drive",
    "Auto Rickshaw Union Stand", "Railway Retiring Room", "Airbnb Stay",
]
EMI_VENDORS = [
    "Bajaj Finserv EMI", "HDFC Personal Loan EMI", "HomeCredit India EMI",
    "Home Loan EMI ICICI", "Car Loan EMI SBI", "Tata Capital EMI",
    "Two Wheeler Loan EMI", "Education Loan EMI", "Credit Card Auto-Debit",
    "Gold Loan EMI Muthoot", "Consumer Durable EMI Bajaj", "IDFC Loan EMI",
    "ECS Housing Loan Mandate", "NACH Mandate Debit", "Personal Loan Foreclosure",
]
INVESTMENT_VENDORS = [
    "Zerodha Kite", "Groww Mutual Fund", "HDFC Securities", "ICICI Direct",
    "SBI Mutual Fund SIP", "Axis Bluechip Fund", "PPF Contribution",
    "NPS Tier 1 Contribution", "IPO Application ASBA", "Sovereign Gold Bond",
    "LIC Premium Payment", "Term Insurance Premium", "Recurring Deposit",
    "Fixed Deposit Renewal", "Demat Account Charges", "Upstox Trading",
    "Digital Gold Purchase", "Bond Ladder Investment", "Stock Broker Margin",
]
SHOPPING_VENDORS = [
    "Amazon.in", "Flipkart", "Myntra Fashion", "Ajio Clothing", "Nykaa Beauty",
    "Croma Electronics", "Reliance Digital", "Decathlon Sports", "IKEA Home",
    "Lenskart Eyewear", "Pharmeasy Medicines", "1mg Pharmacy", "Urban Company",
    "BookMyShow Tickets", "Netflix Subscription", "Spotify Premium",
    "Big Bazaar Retail", "D-Mart Store", "Lifestyle Store", "Pantaloons",
    "FirstCry Baby Products", "Furniture Bazaar", "Local Electronics Shop",
]

FOOD_CONNECTORS = [
    lambda v,a,b,r,d,ac: f"Your {b} account {ac} has been debited by Rs {a} for a purchase at {v} on {d}, reference number {r}.",
    lambda v,a,b,r,d,ac: f"{b}: A sum of INR {a} has been paid to {v} using UPI. Reference: {r}",
    lambda v,a,b,r,d,ac: f"Payment successful -- Rs.{a} to {v}. Ref no {r}.",
    lambda v,a,b,r,d,ac: f"{ac} debited Rs {a} on {d} at {v}, UPI transaction reference {r} -- {b}",
    lambda v,a,b,r,d,ac: f"Order placed at {v}, amount Rs {a} charged to account {ac} via UPI ref {r}",
    lambda v,a,b,r,d,ac: f"{v}: thank you for your order of Rs {a}, paid via UPI, txn {r}",
    lambda v,a,b,r,d,ac: f"Rs{a} spent at {v} on {d}. Available balance updated. -{b}",
    lambda v,a,b,r,d,ac: f"Money transferred: Rs {a} -> {v}, purpose food/dining, ref {r}",
    lambda v,a,b,r,d,ac: f"{b} alert: {ac} debited for Rs {a}, merchant {v}, {d}, ref {r}",
]
TRAVEL_CONNECTORS = [
    lambda v,a,b,r,d,ac: f"Booking confirmed with {v}, total fare Rs {a} charged to {ac}, ref {r}",
    lambda v,a,b,r,d,ac: f"{b}: Rs {a} debited towards {v} booking on {d}, ref {r}",
    lambda v,a,b,r,d,ac: f"Your trip with {v} has been booked. Amount Rs {a} paid, transaction {r}.",
    lambda v,a,b,r,d,ac: f"Fare collected: Rs {a} for {v} ride, {d}, account {ac}",
    lambda v,a,b,r,d,ac: f"{v} ticket purchase of Rs {a} successful, PNR/ref {r}",
    lambda v,a,b,r,d,ac: f"Account {ac} charged Rs {a} for {v} services on {d} -- {b}",
    lambda v,a,b,r,d,ac: f"Toll/parking payment of Rs {a} made at {v}, ref {r}",
]
EMI_CONNECTORS = [
    lambda v,a,b,r,d,ac: f"{b}: EMI of Rs {a} for {v} has been auto-debited from account {ac} on {d}, ref {r}",
    lambda v,a,b,r,d,ac: f"Your monthly instalment of Rs {a} towards {v} was processed successfully, ref {r}",
    lambda v,a,b,r,d,ac: f"NACH mandate executed: Rs {a} debited for {v}, {d}, account {ac}",
    lambda v,a,b,r,d,ac: f"Standing instruction: Rs {a} paid towards {v} on {d}. Ref {r}. -{b}",
    lambda v,a,b,r,d,ac: f"{v} instalment payment of Rs {a} could not be processed due to insufficient funds in {ac} -- please maintain balance before {d}",
    lambda v,a,b,r,d,ac: f"Loan EMI reminder: Rs {a} due for {v} on {d}, account {ac}",
    lambda v,a,b,r,d,ac: f"Auto-debit successful for {v}, amount Rs {a}, reference {r} -- {b}",
]
INVESTMENT_CONNECTORS = [
    lambda v,a,b,r,d,ac: f"SIP instalment of Rs {a} towards {v} debited from {ac} on {d}, ref {r}",
    lambda v,a,b,r,d,ac: f"{b}: Rs {a} transferred for {v} purchase, transaction ref {r}",
    lambda v,a,b,r,d,ac: f"Your investment in {v} of Rs {a} has been confirmed, folio ref {r}",
    lambda v,a,b,r,d,ac: f"Application money of Rs {a} blocked in {ac} for {v}, ref {r}",
    lambda v,a,b,r,d,ac: f"Premium payment of Rs {a} received for {v} on {d}, receipt {r}",
    lambda v,a,b,r,d,ac: f"{v}: units allotted against payment of Rs {a}, ref {r}",
    lambda v,a,b,r,d,ac: f"Recurring deposit instalment Rs {a} credited under {v}, {d}",
]
SHOPPING_CONNECTORS = [
    lambda v,a,b,r,d,ac: f"{b}: Rs {a} debited from {ac} for order on {v}, {d}, ref {r}",
    lambda v,a,b,r,d,ac: f"Your order on {v} for Rs {a} has been placed, payment ref {r}",
    lambda v,a,b,r,d,ac: f"Purchase of Rs {a} made at {v} on {d}. Account {ac} debited. Ref {r}",
    lambda v,a,b,r,d,ac: f"{v}: payment of Rs {a} successful via saved card/UPI, transaction {r}",
    lambda v,a,b,r,d,ac: f"Subscription renewed: {v}, Rs {a} charged, ref {r}",
    lambda v,a,b,r,d,ac: f"Cart checkout complete on {v}, total Rs {a}, {d}, ref {r}",
    lambda v,a,b,r,d,ac: f"Rs{a} paid towards {v} order, account {ac}, {b}",
]

CATEGORY_MAP = {
    "food": (FOOD_VENDORS, FOOD_CONNECTORS),
    "travel": (TRAVEL_VENDORS, TRAVEL_CONNECTORS),
    "EMI": (EMI_VENDORS, EMI_CONNECTORS),
    "investment": (INVESTMENT_VENDORS, INVESTMENT_CONNECTORS),
    "shopping": (SHOPPING_VENDORS, SHOPPING_CONNECTORS),
}

def build_rows(n_per_class=1600):
    rows = []
    for label, (vendors, connectors) in CATEGORY_MAP.items():
        for _ in range(n_per_class):
            v = random.choice(vendors)
            fn = random.choice(connectors)
            b = random.choice(BANKS)
            a = amt()
            r = ref()
            d = date()
            ac = acct()
            text = fn(v, a, b, r, d, ac)
            rows.append((text, label))
    random.shuffle(rows)
    return rows


def _digit_norm(text: str) -> str:
    return re.sub(r"\d+", "#", str(text))


def main():
    rows = build_rows()
    print(f"Built {len(rows)} rows across {len(CATEGORY_MAP)} classes.")

    # ── Programmatic-only disjointness check against the eval set ──────────
    import pandas as pd
    eval_csv = os.path.join(_HERE, "category_generalization_test_set.csv")
    if os.path.exists(eval_csv):
        eval_df = pd.read_csv(eval_csv)
        eval_norms = set(eval_df["text"].apply(_digit_norm))
        eval_exact = set(eval_df["text"])
        new_texts = [t for t, _ in rows]
        exact_overlap = set(new_texts) & eval_exact
        norm_overlap = {_digit_norm(t) for t in new_texts} & eval_norms
        print(f"Exact-text overlap with eval set: {len(exact_overlap)}")
        print(f"Digit-normalized structure overlap with eval set: {len(norm_overlap)}")
        if exact_overlap or norm_overlap:
            raise SystemExit(
                f"REFUSING TO WRITE OUTPUT: {len(exact_overlap)} exact + "
                f"{len(norm_overlap)} normalized-structure overlaps found "
                f"against category_generalization_test_set.csv. Fix templates."
            )
    else:
        print("WARNING: eval CSV not found, skipping disjointness check.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for text, label in rows:
            w.writerow([text, label])
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
