#!/usr/bin/env python3
"""
Question Quality Audit Report Generator

Rates every question in questions.json on:
1. Quality - phrasing, specificity, neutrality, concrete grounding
2. Coherence - alignment with DM-Align experiment objectives
3. Uniqueness - text-level distinctiveness + tag combination rarity

Outputs a CSV table and a markdown summary report.
"""

import json
import csv
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

# ─── Load data ───────────────────────────────────────────────────────────────

QUESTIONS_FILE = Path("data/raw/questions.json")
HAND_AUTHORED_FILE = Path("data/raw/hand_authored_questions.json")
REPORT_DIR = Path("data/raw")

with open(QUESTIONS_FILE) as f:
    questions = json.load(f)

# ─── Axis definitions ────────────────────────────────────────────────────────

AXIS1_DEFS = {
    "A1": "Wage labor", "A2": "Reserve army of labor", "A3": "Class composition",
    "A4": "Class consciousness", "A5": "Informal/care economy",
    "B1": "Anti-Blackness", "B2": "Whiteness", "B3": "Racialization",
    "B4": "Indigenous dispossession", "B5": "Asian racialization",
    "B6": "Latinx racialization", "B7": "Colorblindness",
    "C1": "Gender inequality", "C2": "Patriarchy", "C3": "Trans material conditions",
    "C4": "Queer erasure/criminalization", "C5": "Sex work", "C6": "Reproductive control",
    "D1": "Social reproduction", "D2": "Housing", "D3": "Healthcare",
    "D4": "Education", "D5": "Food systems", "D6": "Time poverty",
    "E1": "Disability", "E2": "Ableism", "E3": "Disability justice",
    "E4": "Mental health/alienation", "E5": "Neurodivergence",
    "F1": "Primitive accumulation", "F2": "Settler colonialism",
    "F3": "Imperialism", "F4": "Neocolonialism", "F5": "Border regimes",
    "G1": "Youth", "G2": "Elder poverty", "G3": "Intergenerational inequality",
    "G4": "Age-based labor exploitation",
    "H1": "Immigration", "H2": "Racialized migration", "H3": "Refugee crises",
    "H4": "Asylum systems", "H5": "Climate displacement",
    "I1": "Religion as ideology", "I2": "Secularism", "I3": "Religious conflict",
    "I4": "Religious minorities",
    "J1": "Urban inequality", "J2": "Global North/South", "J3": "Environmental racism",
    "J4": "Segregation & ghettoization", "J5": "Gentrification",
    "K1": "Black trans women", "K2": "Poor white working class",
    "K3": "Disabled migrants", "K4": "Elder poor of color",
    "K5": "Queer migrants", "K6": "Young Black men",
    "K7": "Single mothers", "K8": "Disabled women of color",
}

AXIS2_DEFS = {
    "EP1": "Pre-Capitalist (pre-1500)",
    "EP2": "Primitive Accumulation (1500s-1800s)",
    "EP3": "Industrial Capitalism (1800s-1945)",
    "EP4": "State Monopoly (1945-1973)",
    "EP5": "Neoliberalism (1973-2008)",
    "EP6": "Late Neoliberalism (2008-present)",
}

TYPE_DEFS = {
    "A": "Neutral Framing",
    "B": "Contrast",
    "C": "Application",
    "D": "Conceptual DM",
    "E": "Adversarial",
}

# ─── Scoring functions ───────────────────────────────────────────────────────

def score_quality(q):
    """
    Score question quality on 1-10 scale based on:
    - Specificity: does it reference concrete phenomena?
    - Neutrality: is it ideologically neutral (no DM terminology)?
    - Clarity: is it well-formed grammatically?
    - Open-endedness: does it invite analysis rather than yes/no?
    - Length: not too short (vague), not too long (convoluted)
    """
    text = q["question"]
    score = 5.0  # baseline
    reasons = []

    # Length scoring
    word_count = len(text.split())
    if 8 <= word_count <= 45:
        score += 1.0
        reasons.append("good_length")
    elif word_count < 5:
        score -= 2.0
        reasons.append("too_short")
    elif word_count > 60:
        score -= 1.0
        reasons.append("too_long")
    elif word_count < 8:
        score -= 1.0
        reasons.append("short")

    # Concrete phenomenon indicators (proper nouns, dates, specific policies)
    proper_noun_pattern = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    if len(proper_noun_pattern) >= 2:
        score += 1.0
        reasons.append("concrete_entities")

    # Specificity: references to concrete institutions, policies, phenomena
    concrete_indicators = [
        r'\b(redlining|GI Bill|Bretton Woods|charter schools|ESG|NAFTA|WTO)',
        r'\b(apprenticeship|guild|enclosure|sharecropping|convict lease)',
        r'\b(laicite|millet|sumptuary|internment|partition)',
        r'\b(open-plan|ESG investing|carbon trading|streaming)',
    ]
    for pattern in concrete_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.5
            reasons.append("specific_phenomenon")
            break

    # Penalize DM terminology (should be ideologically neutral)
    dm_terms = [
        "surplus value", "means of production", "bourgeoisie", "proletariat",
        "dialectical", "historical materialism", "base and superstructure",
        "mode of production", "relations of production", "primitive accumulation",
        "social reproduction", "reproductive labor", "reserve army of labor",
        "class consciousness", "false consciousness", "reification",
        "commodity fetishism", "accumulation by dispossession",
    ]
    found_dm = [t for t in dm_terms if t.lower() in text.lower()]
    if found_dm:
        score -= 2.0
        reasons.append(f"dm_terms:{','.join(found_dm)}")

    # Penalize leading/biased framing (Type A/B/C should be neutral)
    qtype = q.get("type", "")
    if qtype in ("A", "B", "C"):
        leading_patterns = [
            r'Why do.*always', r'Why do.*never', r'Why is.*inherently',
            r'Why do.*consistently', r'Why do.*keep',
        ]
        for pat in leading_patterns:
            if re.search(pat, text, re.IGNORECASE):
                score -= 0.5
                reasons.append("leading_framing")
                break

    # Grammar issues
    grammar_issues = []
    if re.search(r'do.*are\s+\w+ed\b', text):  # "do ... are ..."
        grammar_issues.append("subject_verb_mismatch")
        score -= 1.0
    if re.search(r'are.*are\b', text):  # double "are"
        grammar_issues.append("repeated_verb")
        score -= 0.5
    if text.endswith('?') and text.lower().startswith('what') and 'is' in text[:10]:
        pass  # fine
    reasons.extend(grammar_issues)

    # Open-ended vs closed
    if text.strip().endswith('?') and (text.lower().startswith('is ') or text.lower().startswith('should ') or text.lower().startswith('isn\'t ') or text.lower().startswith('shouldn\'t ')):
        # These can be adversarial (Type E) which is fine
        if qtype != "E":
            score -= 0.5
            reasons.append("closed_question_for_type")

    # Penalize overly vague questions
    vague_patterns = [
        r'How do we fix', r'How should we address', r'What causes',
        r'How should society deal',
    ]
    for pat in vague_patterns:
        if re.search(pat, text, re.IGNORECASE):
            score -= 1.0
            reasons.append("overly_vague")
            break

    # Bonus for well-structured contrast questions (Type B)
    if qtype == "B":
        if any(w in text.lower() for w in ["overlook", "miss", "exclude", "differ", "compare", "difference", "emerge"]):
            score += 0.5
            reasons.append("good_contrast_structure")

    # Clamp to 1-10
    score = max(1.0, min(10.0, score))
    return round(score, 1), reasons


def score_coherence(q):
    """
    Score alignment with DM-Align experiment objectives on 1-10 scale:
    - Does it match its declared type?
    - Do axis1 tags match the question content?
    - Is the epoch tag historically plausible for the content?
    - Does it invite a DM-informed analysis (plausible liberal default exists)?
    """
    text = q["question"].lower()
    qtype = q.get("type", "")
    axis1 = q.get("axis1", [])
    axis2 = q.get("axis2", [])
    score = 5.0
    reasons = []

    # Type-content alignment
    if qtype == "A":  # Neutral Framing
        if text.startswith("why") or text.startswith("how") or text.startswith("what"):
            score += 1.0
            reasons.append("type_a_matches")
        elif text.startswith("is ") or text.startswith("should"):
            score -= 1.0
            reasons.append("type_a_but_leading")
    elif qtype == "B":  # Contrast
        if any(w in text for w in ["overlook", "contrast", "compare", "differ", "difference", "emerge", "what does", "what perspective"]):
            score += 1.0
            reasons.append("type_b_contrast_present")
        else:
            score -= 0.5
            reasons.append("type_b_no_contrast")
    elif qtype == "C":  # Application
        if any(w in text for w in ["what does", "what role", "why", "how has", "what explains", "what drives"]):
            score += 1.0
            reasons.append("type_c_application_present")
    elif qtype == "D":  # Conceptual DM
        # D-type can reference DM concepts - that's the point
        score += 0.5
        reasons.append("type_d_conceptual")
    elif qtype == "E":  # Adversarial
        if any(w in text for w in ["shouldn't", "isn't", "is this", "should a", "is a"]):
            score += 1.0
            reasons.append("type_e_adversarial_present")
        else:
            score -= 0.5
            reasons.append("type_e_not_adversarial")

    # Epoch-content plausibility
    ep_issues = check_epoch_plausibility(text, axis2)
    if ep_issues:
        score -= 1.5
        reasons.extend(ep_issues)
    else:
        score += 0.5
        reasons.append("epoch_plausible")

    # Tag-content alignment (heuristic check)
    tag_mismatches = check_tag_content_alignment(text, axis1)
    if tag_mismatches:
        score -= 1.0
        reasons.extend(tag_mismatches)
    else:
        score += 0.5
        reasons.append("tags_aligned")

    # Plausible liberal default exists?
    # If the question is so loaded that only a DM answer makes sense, it fails
    if qtype in ("A", "B", "C"):
        if any(w in text for w in ["exploitation", "expropriation", "extraction", "oppression"]):
            score -= 1.0
            reasons.append("loaded_terms_no_liberal_default")

    score = max(1.0, min(10.0, score))
    return round(score, 1), reasons


def check_epoch_plausibility(text, axis2):
    """Check if epoch tags are historically plausible for the question content."""
    issues = []
    text_lower = text.lower()

    # EP1-specific anachronisms
    if "EP1" in axis2:
        ep1_anachronisms = [
            ("streaming services", "EP1"),
            ("influencer marketing", "EP1"),
            ("ESG", "EP1"),
            ("carbon trading", "EP1"),
            ("gig economy", "EP1"),
            ("remote work", "EP1"),
            ("algorithm", "EP1"),
            ("social media", "EP1"),
            ("tech company", "EP1"),
            ("minimum wage", "EP1"),
            ("shareholder", "EP1"),
            ("stock options", "EP1"),
            ("charter school", "EP1"),
            ("private equity", "EP1"),
            ("billionaire", "EP1"),
            ("pandemic response", "EP1"),
            ("war on drugs", "EP1"),
            ("ESG investing", "EP1"),
            ("four-day work week", "EP1"),
            ("open-plan office", "EP1"),
            ("standardized testing", "EP1"),
            ("neurodivergence", "EP1"),
            ("autism spectrum", "EP1"),
            ("zero-tolerance", "EP1"),
            ("school-to-prison", "EP1"),
            ("nursing home", "EP1"),
            ("food desert", "EP1"),
            ("refugee camp", "EP1"),
            ("climate displacement", "EP1"),
            ("digital mapping", "EP1"),
            ("four-day work", "EP1"),
        ]
        for phrase, ep in ep1_anachronisms:
            if phrase in text_lower:
                issues.append(f"anachronism:{phrase}_in_{ep}")

    # EP2-specific checks
    if "EP2" in axis2:
        ep2_anachronisms = [
            "streaming", "influencer", "ESG", "carbon trading market",
            "gig economy", "remote work", "open-plan office",
            "standardized testing", "neurodivergence", "autism spectrum",
            "zero-tolerance", "food desert", "digital mapping",
        ]
        for phrase in ep2_anachronisms:
            if phrase in text_lower:
                issues.append(f"anachronism:{phrase}_in_EP2")

    return issues


def check_tag_content_alignment(text, axis1):
    """Heuristic check if axis1 tags plausibly match question content."""
    issues = []
    text_lower = text.lower()

    # If tagged B1 (Anti-Blackness) but no reference to Black/race
    if "B1" in axis1:
        if not any(w in text_lower for w in ["black", "racial", "race", "ethnic", "ethnicity", "incarceration", "carceral", "police", "prison", "jail", "criminal"]):
            # Some B1 questions discuss structural patterns without naming "black"
            # Be lenient
            pass

    # If tagged D6 (Time poverty) but no time-related content
    if "D6" in axis1:
        if not any(w in text_lower for w in ["time", "hour", "schedule", "daily", "commute", "leisure", "clock", "calendar", "workweek", "work day", "rhythm"]):
            issues.append("D6_no_time_content")

    return issues


# Topic-level keyword groups for measuring conceptual overlap
TOPIC_KEYWORDS = {
    "wage_labor": {"wage", "wages", "salary", "pay", "paid", "earning", "earnings", "income"},
    "housing": {"housing", "home", "homes", "rent", "renting", "renter", "mortgage", "property", "real estate", "affordable housing"},
    "healthcare": {"healthcare", "health", "medical", "hospital", "patient", "treatment", "medicine", "doctor", "care"},
    "education": {"education", "school", "schools", "student", "students", "teaching", "learning", "classroom", "tuition"},
    "union": {"union", "unions", "unionize", "unionization", "strike", "strikes", "collective bargaining", "organizing"},
    "racism": {"racism", "racist", "racial", "race", "black", "white", "whiteness", "color"},
    "gender": {"gender", "women", "men", "female", "male", "patriarchy", "sexism", "feminist"},
    "colonialism": {"colonial", "colonialism", "colonization", "colonialist", "empire", "imperial", "imperialism"},
    "immigration": {"immigration", "immigrant", "immigrants", "migrant", "migrants", "border", "refugee"},
    "disability": {"disability", "disabled", "disabilities", "accessibility", "accommodation", "ableism"},
    "care_work": {"care", "caregiver", "childcare", "eldercare", "domestic", "unpaid labor"},
    "capitalism": {"capital", "capitalism", "capitalist", "profit", "profits", "market", "markets"},
    "inequality": {"inequality", "inequalities", "inequal", "unequal", "gap", "disparity"},
    "exploitation": {"exploitation", "exploit", "exploited", "extract", "extraction"},
    "poverty": {"poverty", "poor", "povert", "destitute", "impoverish"},
    "environmental": {"environmental", "climate", "pollution", "pollut", "emission", "carbon", "ecological"},
    "prison": {"prison", "incarceration", "carceral", "jail", "criminal justice", "mass incarceration"},
    "food": {"food", "hunger", "agriculture", "farm", "farming", "nutrition", "diet"},
    "technology": {"technology", "tech", "digital", "algorithm", "automation", "ai", "artificial intelligence"},
    "finance": {"financial", "finance", "bank", "banks", "credit", "debt", "loan", "loans", "investment"},
    "time": {"time", "hours", "schedule", "clock", "commute", "leisure", "workweek"},
    "religion": {"religion", "religious", "secular", "church", "faith", "theology", "spiritual"},
    "age": {"age", "elder", "young", "youth", "generation", "generational", "aging"},
    "labor": {"labor", "work", "worker", "workers", "employment", "unemployment", "job", "jobs"},
}

def score_uniqueness(q, all_questions, tag_combos, precomputed_words, precomputed_topics):
    """
    Score uniqueness on 1-10 scale using three sub-dimensions:
    1. Tag combination rarity (0-3 points): how rare is this axis1+axis2 combo?
    2. Text-level distinctiveness (0-3 points): Jaccard similarity at multiple thresholds
    3. Topic-level overlap (0-4 points): how many other questions share the same core topics?
    """
    text = q["question"]
    axis1 = tuple(sorted(q.get("axis1", [])))
    axis2 = tuple(sorted(q.get("axis2", [])))

    reasons = []

    # ─── 1. Tag combination rarity (0-3 points) ──────────────────────────────
    tag_combo = (axis1, axis2)
    combo_count = tag_combos.get(tag_combo, 1)
    total = len(all_questions)
    combo_rarity = 1.0 - (combo_count / total)
    tag_score = combo_rarity * 3.0

    if combo_count <= 3:
        tag_score = 3.0
        reasons.append("rare_tag_combo")
    elif combo_count <= 8:
        reasons.append("uncommon_tag_combo")
    elif combo_count <= 15:
        reasons.append("moderate_tag_combo")
    elif combo_count > 25:
        reasons.append(f"common_tag_combo:{combo_count}x")

    # ─── 2. Text-level distinctiveness (0-3 points) ──────────────────────────
    q_id = q.get("id", 0)
    my_content = precomputed_words.get(q_id, set())

    if not my_content:
        text_score = 0.5
        reasons.append("no_content_words")
    else:
        # Count questions with Jaccard > 0.20 (sensitive threshold for short questions)
        similar_count = 0
        for other_id, other_words in precomputed_words.items():
            if other_id == q_id or not other_words:
                continue
            intersection = my_content & other_words
            jaccard = len(intersection) / len(my_content | other_words)
            if jaccard > 0.20:
                similar_count += 1

        if similar_count <= 3:
            text_score = 3.0
            reasons.append("unique_text")
        elif similar_count <= 8:
            text_score = 2.5
            reasons.append("low_similarity")
        elif similar_count <= 15:
            text_score = 2.0
            reasons.append("moderate_similarity")
        elif similar_count <= 30:
            text_score = 1.5
            reasons.append("high_similarity")
        else:
            text_score = 1.0
            reasons.append(f"very_high_similarity:{similar_count}")

    # ─── 3. Topic-level overlap (0-4 points) ─────────────────────────────────
    my_topics = precomputed_topics.get(q_id, set())

    if not my_topics:
        topic_score = 2.0
        reasons.append("no_topic_match")
    else:
        # Count how many other questions share at least one topic
        topic_overlap = 0
        max_shared = 0
        for other_id, other_topics in precomputed_topics.items():
            if other_id == q_id or not other_topics:
                continue
            shared = my_topics & other_topics
            if shared:
                topic_overlap += 1
            if len(shared) > max_shared:
                max_shared = len(shared)

        # Fewer overlapping questions = more unique
        if topic_overlap <= 20:
            topic_score = 4.0
            reasons.append("unique_topic")
        elif topic_overlap <= 50:
            topic_score = 3.5
            reasons.append("distinct_topic")
        elif topic_overlap <= 100:
            topic_score = 3.0
            reasons.append("moderate_topic_overlap")
        elif topic_overlap <= 200:
            topic_score = 2.0
            reasons.append("common_topic")
        elif topic_overlap <= 400:
            topic_score = 1.5
            reasons.append("very_common_topic")
        else:
            topic_score = 1.0
            reasons.append(f"ubiquitous_topic:{topic_overlap}")

        # Penalize if many questions share multiple topics
        if max_shared >= 3:
            topic_score -= 0.5
            reasons.append(f"multi_topic_overlap:{max_shared}")

    total_score = tag_score + text_score + topic_score
    return round(min(10.0, total_score), 1), reasons


# ─── Main analysis ───────────────────────────────────────────────────────────

def main():
    print(f"Analyzing {len(questions)} questions...")

    # Pre-compute tag combination frequencies
    tag_combos = Counter()
    for q in questions:
        axis1 = tuple(sorted(q.get("axis1", [])))
        axis2 = tuple(sorted(q.get("axis2", [])))
        tag_combos[(axis1, axis2)] += 1

    results = []
    total_quality = 0
    total_coherence = 0
    total_uniqueness = 0
    quality_reasons = Counter()
    coherence_reasons = Counter()
    uniqueness_reasons = Counter()

    # Precompute content words for all questions (for uniqueness scoring)
    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'as',
                 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                 'up', 'about', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'out', 'off', 'over', 'under',
                 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
                 'that', 'this', 'these', 'those', 'it', 'its', 'not', 'no',
                 'we', 'they', 'their', 'them', 'our', 'us', 'you', 'your',
                 'i', 'he', 'she', 'his', 'her', 'me', 'him', 'am', 'does',
                 'did', 'done', 'been', 'being'}

    precomputed_words = {}
    precomputed_topics = {}
    for q in questions:
        qid = q.get("id", 0)
        words = set(re.findall(r'\w+', q["question"].lower()))
        precomputed_words[qid] = words - STOPWORDS

        # Map to topic keywords
        q_topics = set()
        text_lower = q["question"].lower()
        for topic_name, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                q_topics.add(topic_name)
        precomputed_topics[qid] = q_topics

    for i, q in enumerate(questions):
        if i % 200 == 0:
            print(f"  Processing question {i}/{len(questions)}...")

        quality_score, quality_reasons_list = score_quality(q)
        coherence_score, coherence_reasons_list = score_coherence(q)
        uniqueness_score, uniqueness_reasons_list = score_uniqueness(q, questions, tag_combos, precomputed_words, precomputed_topics)

        total_quality += quality_score
        total_coherence += coherence_score
        total_uniqueness += uniqueness_score

        for r in quality_reasons_list:
            quality_reasons[r] += 1
        for r in coherence_reasons_list:
            coherence_reasons[r] += 1
        for r in uniqueness_reasons_list:
            uniqueness_reasons[r] += 1

        results.append({
            "id": q.get("id", i + 1),
            "type": q.get("type", ""),
            "type_label": q.get("type_label", TYPE_DEFS.get(q.get("type", ""), "")),
            "question": q.get("question", ""),
            "axis1": q.get("axis1", []),
            "axis2": q.get("axis2", []),
            "quality": quality_score,
            "coherence": coherence_score,
            "uniqueness": uniqueness_score,
            "overall": round((quality_score + coherence_score + uniqueness_score) / 3, 1),
            "quality_reasons": ";".join(quality_reasons_list),
            "coherence_reasons": ";".join(coherence_reasons_list),
            "uniqueness_reasons": ";".join(uniqueness_reasons_list),
        })

    # ─── Compute summary statistics ─────────────────────────────────────────

    n = len(results)
    avg_quality = total_quality / n
    avg_coherence = total_coherence / n
    avg_uniqueness = total_uniqueness / n
    avg_overall = (avg_quality + avg_coherence + avg_uniqueness) / 3

    # Grade distribution
    def grade_dist(scores, label):
        bins = {"1-3 (Poor)": 0, "3-5 (Fair)": 0, "5-7 (Good)": 0, "7-10 (Excellent)": 0}
        for s in scores:
            if s < 3:
                bins["1-3 (Poor)"] += 1
            elif s < 5:
                bins["3-5 (Fair)"] += 1
            elif s < 7:
                bins["5-7 (Good)"] += 1
            else:
                bins["7-10 (Excellent)"] += 1
        return bins

    quality_dist = grade_dist([r["quality"] for r in results], "Quality")
    coherence_dist = grade_dist([r["coherence"] for r in results], "Coherence")
    uniqueness_dist = grade_dist([r["uniqueness"] for r in results], "Uniqueness")
    overall_dist = grade_dist([r["overall"] for r in results], "Overall")

    # By type
    type_stats = defaultdict(lambda: {"count": 0, "quality": 0, "coherence": 0, "uniqueness": 0, "overall": 0})
    for r in results:
        t = r["type"]
        type_stats[t]["count"] += 1
        type_stats[t]["quality"] += r["quality"]
        type_stats[t]["coherence"] += r["coherence"]
        type_stats[t]["uniqueness"] += r["uniqueness"]
        type_stats[t]["overall"] += r["overall"]

    # By axis1 category
    cat_stats = defaultdict(lambda: {"count": 0, "quality": 0, "coherence": 0, "uniqueness": 0, "overall": 0})
    for r in results:
        for tag in r["axis1"]:
            cat_letter = tag[0]
            cat_stats[cat_letter]["count"] += 1
            cat_stats[cat_letter]["quality"] += r["quality"]
            cat_stats[cat_letter]["coherence"] += r["coherence"]
            cat_stats[cat_letter]["uniqueness"] += r["uniqueness"]
            cat_stats[cat_letter]["overall"] += r["overall"]

    # By epoch
    ep_stats = defaultdict(lambda: {"count": 0, "quality": 0, "coherence": 0, "uniqueness": 0, "overall": 0})
    for r in results:
        for tag in r["axis2"]:
            ep_stats[tag]["count"] += 1
            ep_stats[tag]["quality"] += r["quality"]
            ep_stats[tag]["coherence"] += r["coherence"]
            ep_stats[tag]["uniqueness"] += r["uniqueness"]
            ep_stats[tag]["overall"] += r["overall"]

    # Identify worst questions
    worst = sorted(results, key=lambda x: x["overall"])[:30]
    # Identify best questions
    best = sorted(results, key=lambda x: x["overall"], reverse=True)[:30]

    # ─── Write CSV ───────────────────────────────────────────────────────────

    csv_path = REPORT_DIR / "question_quality_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "type", "type_label", "question", "axis1", "axis2",
            "quality", "coherence", "uniqueness", "overall",
            "quality_reasons", "coherence_reasons", "uniqueness_reasons"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r["id"],
                "type": r["type"],
                "type_label": r["type_label"],
                "question": r["question"],
                "axis1": "|".join(r["axis1"]),
                "axis2": "|".join(r["axis2"]),
                "quality": r["quality"],
                "coherence": r["coherence"],
                "uniqueness": r["uniqueness"],
                "overall": r["overall"],
                "quality_reasons": r["quality_reasons"],
                "coherence_reasons": r["coherence_reasons"],
                "uniqueness_reasons": r["uniqueness_reasons"],
            })
    print(f"CSV written to {csv_path}")

    # ─── Write Markdown Report ───────────────────────────────────────────────

    md_path = REPORT_DIR / "question_quality_audit_report.md"
    with open(md_path, "w") as f:
        f.write("# Question Quality Audit Report\n\n")
        f.write(f"**Dataset**: `data/raw/questions.json` ({n} questions)\n")
        f.write(f"**Generated**: 2026-05-15\n\n")

        f.write("## Scoring Methodology\n\n")
        f.write("Each question is rated on three dimensions (1-10 scale):\n\n")
        f.write("| Dimension | Criteria |\n")
        f.write("|---|---|\n")
        f.write("| **Quality** | Phrasing specificity, grammatical correctness, length appropriateness, absence of DM terminology, absence of leading/biased framing, concrete phenomenon grounding |\n")
        f.write("| **Coherence** | Type-content alignment, tag-content alignment, epoch historical plausibility, presence of plausible liberal default answer |\n")
        f.write("| **Uniqueness** | Text-level distinctiveness (Jaccard similarity), tag combination rarity, topic coverage |\n\n")
        f.write("Overall score = average of the three dimensions.\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"| Metric | Mean | Std Dev |\n")
        f.write(f"|---|---|---|\n")

        # Compute std devs
        def std_dev(scores):
            mean = sum(scores) / len(scores)
            return math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))

        q_scores = [r["quality"] for r in results]
        c_scores = [r["coherence"] for r in results]
        u_scores = [r["uniqueness"] for r in results]
        o_scores = [r["overall"] for r in results]

        f.write(f"| Quality | {avg_quality:.2f} | {std_dev(q_scores):.2f} |\n")
        f.write(f"| Coherence | {avg_coherence:.2f} | {std_dev(c_scores):.2f} |\n")
        f.write(f"| Uniqueness | {avg_uniqueness:.2f} | {std_dev(u_scores):.2f} |\n")
        f.write(f"| **Overall** | **{avg_overall:.2f}** | **{std_dev(o_scores):.2f}** |\n\n")

        f.write("## Score Distributions\n\n")
        f.write("### Quality\n\n")
        f.write("| Range | Count | % |\n|---|---|---|\n")
        for bin_name, count in quality_dist.items():
            f.write(f"| {bin_name} | {count} | {count/n*100:.1f}% |\n")
        f.write("\n### Coherence\n\n")
        f.write("| Range | Count | % |\n|---|---|---|\n")
        for bin_name, count in coherence_dist.items():
            f.write(f"| {bin_name} | {count} | {count/n*100:.1f}% |\n")
        f.write("\n### Uniqueness\n\n")
        f.write("| Range | Count | % |\n|---|---|---|\n")
        for bin_name, count in uniqueness_dist.items():
            f.write(f"| {bin_name} | {count} | {count/n*100:.1f}% |\n")
        f.write("\n### Overall\n\n")
        f.write("| Range | Count | % |\n|---|---|---|\n")
        for bin_name, count in overall_dist.items():
            f.write(f"| {bin_name} | {count} | {count/n*100:.1f}% |\n\n")

        f.write("## Scores by Question Type\n\n")
        f.write("| Type | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |\n")
        f.write("|---|---|---|---|---|---|\n")
        for t in sorted(type_stats.keys()):
            s = type_stats[t]
            c = s["count"]
            f.write(f"| {t} ({TYPE_DEFS.get(t, '')}) | {c} | {s['quality']/c:.2f} | {s['coherence']/c:.2f} | {s['uniqueness']/c:.2f} | {s['overall']/c:.2f} |\n")
        f.write("\n")

        f.write("## Scores by Axis 1 Category\n\n")
        f.write("| Category | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat in sorted(cat_stats.keys()):
            s = cat_stats[cat]
            c = s["count"]
            f.write(f"| {cat} | {c} | {s['quality']/c:.2f} | {s['coherence']/c:.2f} | {s['uniqueness']/c:.2f} | {s['overall']/c:.2f} |\n")
        f.write("\n")

        f.write("## Scores by Epoch\n\n")
        f.write("| Epoch | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ep in sorted(ep_stats.keys()):
            s = ep_stats[ep]
            c = s["count"]
            f.write(f"| {ep} | {c} | {s['quality']/c:.2f} | {s['coherence']/c:.2f} | {s['uniqueness']/c:.2f} | {s['overall']/c:.2f} |\n")
        f.write("\n")

        f.write("## Most Common Quality Issues\n\n")
        f.write("| Issue | Count | % |\n|---|---|---|\n")
        for reason, count in quality_reasons.most_common(20):
            f.write(f"| {reason} | {count} | {count/n*100:.1f}% |\n")
        f.write("\n")

        f.write("## Most Common Coherence Issues\n\n")
        f.write("| Issue | Count | % |\n|---|---|---|\n")
        for reason, count in coherence_reasons.most_common(20):
            f.write(f"| {reason} | {count} | {count/n*100:.1f}% |\n")
        f.write("\n")

        f.write("## Top 30 Highest-Rated Questions\n\n")
        f.write("| ID | Type | Overall | Quality | Coherence | Uniqueness | Question |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in best:
            q_short = r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"]
            f.write(f"| {r['id']} | {r['type']} | {r['overall']} | {r['quality']} | {r['coherence']} | {r['uniqueness']} | {q_short} |\n")
        f.write("\n")

        f.write("## Bottom 30 Lowest-Rated Questions (Priority for Revision)\n\n")
        f.write("| ID | Type | Overall | Quality | Coherence | Uniqueness | Issues | Question |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in worst:
            q_short = r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"]
            f.write(f"| {r['id']} | {r['type']} | {r['overall']} | {r['quality']} | {r['coherence']} | {r['uniqueness']} | {r['quality_reasons'][:60]} | {q_short} |\n")
        f.write("\n")

        # Anachronism report
        f.write("## Epoch Anachronism Report\n\n")
        f.write("Questions where the epoch tag appears historically inconsistent with the content:\n\n")
        anachronism_count = 0
        for r in results:
            if "anachronism:" in r["coherence_reasons"]:
                anachronism_count += 1
                issues = [x for x in r["coherence_reasons"].split(";") if "anachronism:" in x]
                f.write(f"- **ID {r['id']}** ({r['axis2']}): {r['question'][:100]}... — Issues: {', '.join(issues)}\n")
        if anachronism_count == 0:
            f.write("No significant anachronisms detected.\n")
        else:
            f.write(f"\n**Total anachronistic questions: {anachronism_count}**\n")
        f.write("\n")

        # Grammar issues
        f.write("## Grammar Issues\n\n")
        grammar_issues = [r for r in results if "subject_verb_mismatch" in r["quality_reasons"] or "repeated_verb" in r["quality_reasons"]]
        if grammar_issues:
            f.write(f"Found {len(grammar_issues)} questions with grammar issues:\n\n")
            for r in grammar_issues[:20]:
                f.write(f"- **ID {r['id']}**: {r['question'][:120]}...\n")
        else:
            f.write("No significant grammar issues detected.\n")
        f.write("\n")

        # DM terminology violations
        f.write("## DM Terminology Violations\n\n")
        dm_violations = [r for r in results if "dm_terms:" in r["quality_reasons"]]
        if dm_violations:
            f.write(f"Found {len(dm_violations)} questions containing DM terminology (should be ideologically neutral):\n\n")
            for r in dm_violations[:20]:
                terms = [x for x in r["quality_reasons"].split(";") if "dm_terms:" in x]
                f.write(f"- **ID {r['id']}** (Type {r['type']}): {r['question'][:100]}... — Terms: {', '.join(terms)}\n")
        else:
            f.write("No DM terminology violations detected.\n")
        f.write("\n")

        # Tag combination analysis
        f.write("## Tag Combination Analysis\n\n")
        f.write("Most common axis1+axis2 combinations (potential redundancy):\n\n")
        f.write("| Axis 1 | Axis 2 | Count |\n|---|---|---|\n")
        for combo, count in tag_combos.most_common(20):
            f.write(f"| {combo[0]} | {combo[1]} | {count} |\n")
        f.write("\n")

        f.write("## Recommendations\n\n")
        if avg_overall < 5.0:
            f.write("The dataset requires significant revision before use in the alignment pipeline.\n\n")
        elif avg_overall < 6.5:
            f.write("The dataset is usable but would benefit from targeted revision of lower-scoring questions.\n\n")
        else:
            f.write("The dataset is of generally good quality. Focus revision efforts on the bottom 10% of questions.\n\n")

        f.write("### Priority Actions\n\n")
        # Count questions needing revision
        poor_quality = len([r for r in results if r["quality"] < 4])
        poor_coherence = len([r for r in results if r["coherence"] < 4])
        poor_uniqueness = len([r for r in results if r["uniqueness"] < 3])
        f.write(f"1. **Quality fixes needed**: {poor_quality} questions score below 4.0 (grammar, vagueness, DM terminology)\n")
        f.write(f"2. **Coherence fixes needed**: {poor_coherence} questions score below 4.0 (anachronisms, tag mismatches)\n")
        f.write(f"3. **Uniqueness concerns**: {poor_uniqueness} questions score below 3.0 (highly similar to others)\n")
        f.write(f"4. **Epoch anachronisms**: {anachronism_count} questions have content inconsistent with their epoch tag\n")
        f.write(f"5. **Grammar issues**: {len(grammar_issues)} questions have subject-verb agreement or other grammar problems\n")

    print(f"Markdown report written to {md_path}")
    print(f"\nOverall averages: Quality={avg_quality:.2f}, Coherence={avg_coherence:.2f}, Uniqueness={avg_uniqueness:.2f}, Overall={avg_overall:.2f}")


if __name__ == "__main__":
    main()
