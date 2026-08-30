"""Is this the kind of question this assistant exists to answer?

Scope is a property of the assistant's remit, not of what happens to be in the
index today. Nothing here looks at the corpus: "where do I park?" is an internal
company question whether or not anyone has uploaded the parking policy yet. That
separation is what lets the pipeline distinguish "don't ask me this" from "ask me
this, but nobody has written it down" - and the second list is the content
roadmap.

Two corpus-independent checks:

1. `precheck` - the request *form* is wrong. "Write me a poem about the expense
   policy" is on-topic and still not something this assistant does. Regex, free,
   runs before any API call.
2. `classify_domain` - the request *topic* is outside the covered domains.
   Nearest-centroid against two lists of short domain descriptions. Relative, so
   there is no threshold to calibrate.
"""
import re

import numpy as np

from . import llm

REJECTION_MESSAGE = "This question is outside the scope of the available company knowledge."

# --- 1. wrong kind of request -------------------------------------------------

OUT_OF_SCOPE_PATTERNS = [
    (r"\b(weather|temperature|forecast|rain|humidity)\b|อากาศ|ฝนตก|พยากรณ์", "weather"),
    (r"\b(football|soccer|match|tournament|world cup|olympics)\b.*\b(won|win|result|score)\b"
     r"|\bwho won\b|ใครชนะ|ผลบอล", "sports"),
    (r"\b(write|compose|create|generate)\s+(me\s+)?(a|an|some)?\s*"
     r"(poem|song|story|joke|haiku|rap|essay|novel)\b"
     r"|เขียน(กลอน|เพลง|นิยาย|เรียงความ)|แต่งกลอน", "creative_writing"),
    (r"\btell\s+me\s+a\s+(joke|story)\b|เล่า(เรื่องตลก|มุก|นิทาน)|ขำๆ", "creative_writing"),
    (r"\b(stock price|share price|exchange rate|bitcoin|crypto price|market cap)\b"
     r"|ราคาหุ้น|อัตราแลกเปลี่ยน|ราคาทอง", "market_data"),
    (r"\b(what|who|when)\s+(is|are|was)\s+the\s+"
     r"(capital|president|prime minister|population)\b|เมืองหลวง|ประธานาธิบดี", "general_knowledge"),
    (r"\b(recipe|cook|restaurant near|movie|netflix|horoscope)\b"
     r"|สูตรอาหาร|ร้านอาหารแถว|ดูหนัง|ดวงวันนี้", "personal_life"),
]

# --- 2. topic outside the assistant's remit -----------------------------------
#
# These describe the domains the company has decided this assistant covers. They
# are a policy statement, edited when the remit changes - NOT derived from the
# documents in data/. Adding a document must never silently widen the scope.
# Thai terms are included because the users are.

COVERED_DOMAINS = [
    "employee leave, annual leave, sick leave, holidays, absence, carry over, "
    "ลาพักร้อน ลาป่วย วันลา วันหยุด สิทธิ์การลา",
    "expense claims, reimbursement, receipts, per diem, cost centre, "
    "เบิกค่าใช้จ่าย ใบเสร็จ เบิกเงิน ค่าเบี้ยเลี้ยง",
    "business travel, flights, hotel, travel request and approval, "
    "เดินทางเพื่อธุรกิจ จองตั๋ว ที่พัก อนุมัติการเดินทาง",
    "procurement, purchase orders, vendors, quotations, budget approval, "
    "จัดซื้อ จัดจ้าง ผู้ขาย ใบสั่งซื้อ ใบเสนอราคา",
    "IT support, passwords, accounts, system access, laptops, service desk, "
    "ไอที รหัสผ่าน สิทธิ์เข้าใช้งาน คอมพิวเตอร์ แจ้งปัญหา",
    "employee onboarding, new joiner, probation, contract, induction, "
    "พนักงานใหม่ ทดลองงาน สัญญาจ้าง ปฐมนิเทศ",
    "HR policy, payroll, benefits, insurance, working hours, performance review, "
    "นโยบายบุคคล เงินเดือน สวัสดิการ ประกัน เวลาทำงาน ประเมินผล",
    "workplace conduct, dress code, office facilities, parking, desk and seating, "
    "ระเบียบการแต่งกาย เครื่องแบบ สถานที่ทำงาน ที่จอดรถ",
    "internal banking operations, branch procedures, compliance, approval limits, "
    "การปฏิบัติงานภายใน สาขา การกำกับดูแล อำนาจอนุมัติ",
]

OFF_DOMAIN_EXAMPLES = [
    "weather forecast temperature rain tomorrow อากาศ พยากรณ์อากาศ",
    "sports results football match score tournament ผลการแข่งขัน ผลบอล",
    "write a poem song story joke essay for me แต่งกลอน เขียนเพลง เล่าเรื่องตลก",
    "stock price exchange rate crypto gold market ราคาหุ้น ราคาทอง อัตราแลกเปลี่ยน",
    "general world trivia capital city president history population ความรู้ทั่วไป ประวัติศาสตร์",
    "recipes restaurants movies travel for fun horoscope สูตรอาหาร ร้านอาหาร ดูหนัง ดวง",
    "personal advice relationships health diagnosis ปรึกษาปัญหาส่วนตัว สุขภาพ",
]

_centroids = None


def _domain_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Embed the domain descriptions once per process. Independent of the corpus."""
    global _centroids
    if _centroids is None:
        vectors = llm.embed(COVERED_DOMAINS + OFF_DOMAIN_EXAMPLES)
        _centroids = (vectors[:len(COVERED_DOMAINS)], vectors[len(COVERED_DOMAINS):])
    return _centroids


def precheck(query: str) -> str | None:
    """Wrong kind of request. Returns a reason, or None to continue."""
    normalised = " ".join(query.lower().split())
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, normalised):
            return reason
    return None


def classify_domain(query_vector: np.ndarray) -> str | None:
    """Wrong topic. Nearest-centroid, so no threshold needs calibrating."""
    covered, off_domain = _domain_vectors()
    if float(np.max(covered @ query_vector)) >= float(np.max(off_domain @ query_vector)):
        return None
    return "outside_covered_domains"
