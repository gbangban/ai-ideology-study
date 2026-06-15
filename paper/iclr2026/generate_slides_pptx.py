#!/usr/bin/env python3
"""Generate ICLR 2026 spotlight presentation PPTX from paper.

Usage:
    python3 generate_slides_pptx.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn


# --- ICLR Color Scheme ---
PRIMARY = RGBColor(0x7B, 0x2D, 0x8E)   # ICLR purple
ACCENT  = RGBColor(0xF0, 0xAD, 0x00)   # ICLR gold
DARK    = RGBColor(0x33, 0x33, 0x33)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF8, 0xF4, 0xFC)  # Very light purple tint


def set_slide_bg(slide, color):
    """Set slide background color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, text, y=Inches(0.3), height=Inches(0.9)):
    """Add a colored title bar at the top of the slide."""
    txBox = slide.shapes.add_textbox(
        Inches(0), y, Inches(13.333), height
    )
    txBox.fill.solid()
    txBox.fill.fore_color.rgb = PRIMARY
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(0)
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)


def add_bullet_list(slide, items, y=Inches(1.5), x=Inches(0.6), width=Inches(12),
                    font_size=28, bullet_color=DARK, line_spacing=1.4):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(x, y, width, Inches(5))
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = bullet_color
        p.space_after = Pt(8)
        p.level = 0
        # Add bullet
        pPr = p._p.get_or_add_pPr()
        buChar = pPr.makeelement(qn('a:buChar'), {'char': '\u2022'})
        pPr.append(buChar)


def add_block(slide, title, items, y=Inches(1.5), x=Inches(0.6), width=Inches(11.5),
              bg_color=LIGHT_BG, title_color=PRIMARY):
    """Add a colored block with title and bullet items."""
    # Background rectangle
    shape = slide.shapes.add_shape(
        1, x, y, width, Inches(len(items) * 0.55 + 0.8)
    )  # 1 = msoShapeRectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()
    shape.shadow.inherit = False

    # Title
    txBox = slide.shapes.add_textbox(
        x + Inches(0.2), y + Inches(0.1), width - Inches(0.4), Inches(0.5)
    )
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = title_color

    # Items
    txBox2 = slide.shapes.add_textbox(
        x + Inches(0.3), y + Inches(0.6), width - Inches(0.5), Inches(len(items) * 0.5)
    )
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        p.text = item
        p.font.size = Pt(24)
        p.font.color.rgb = DARK
        p.space_after = Pt(4)


def add_notes(slide, text):
    """Add speaker notes."""
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


def add_two_columns(slide, left_items, right_items, left_title="", right_title="",
                    y=Inches(1.5), col_width=Inches(5.8), gap=Inches(0.7)):
    """Add two-column layout with titles and bullet lists."""
    # Left column title
    if left_title:
        txBox = slide.shapes.add_textbox(
            Inches(0.6), y - Inches(0.4), col_width, Inches(0.5)
        )
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = left_title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = PRIMARY

    # Left items
    add_bullet_list(slide, left_items, y, Inches(0.6), col_width, font_size=24)

    # Right column title
    rx = Inches(0.6) + col_width + gap
    if right_title:
        txBox = slide.shapes.add_textbox(rx, y - Inches(0.4), col_width, Inches(0.5))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = right_title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = PRIMARY

    # Right items
    add_bullet_list(slide, right_items, y, rx, col_width, font_size=24)


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ============================================================
    # SLIDE 1: Title
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    set_slide_bg(slide, PRIMARY)

    txBox = slide.shapes.add_textbox(
        Inches(1), Inches(1.5), Inches(11.333), Inches(2.5)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Who's Afraid of Communist AI"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Epistemic Transfer and Ideological Convergence in Language Models"
    p2.font.size = Pt(30)
    p2.font.color.rgb = ACCENT
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    txBox2 = slide.shapes.add_textbox(
        Inches(1), Inches(5), Inches(11.333), Inches(1)
    )
    tf2 = txBox2.text_frame
    p3 = tf2.paragraphs[0]
    p3.text = "Anonymous ICLR Submission  |  ICLR 2026"
    p3.font.size = Pt(24)
    p3.font.color.rgb = WHITE
    p3.alignment = PP_ALIGN.CENTER

    add_notes(slide, "[30 sec] Welcome. Today I'll present our work on how SFT transfers not just vocabulary but deep epistemic priors in language models.")

    # ============================================================
    # SLIDE 2: Problem
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Problem: What Does SFT Really Transfer?")

    add_bullet_list(slide, [
        "SFT is the foundational step in LLM alignment",
        "Alignment literature focuses on safety, helpfulness, preference consistency",
        "Open question: Does SFT transfer reasoning patterns beyond the training domain?",
        "Prior work documents ideological priors as a STATIC property (Kronlund 2024, Lee 2026)",
    ], y=Inches(1.5), font_size=28)

    add_block(slide, "Our Dynamic Question", [
        "Can targeted SFT on a non-dominant analytical framework measurably shift reasoning?",
        "What COLLATERAL EFFECTS does this produce?",
    ], y=Inches(4.5), title_color=ACCENT)

    add_notes(slide, "[45 sec] SFT is how we align models, but we don't understand what it transfers beyond the training domain. Prior work showed LLMs carry ideological priors from pretraining, but treated it as static. We asked: if we fine-tune on a specific analytical framework, does it shift reasoning in unseen domains? Transition: To answer this, we fine-tuned Qwen3.5-9B on Dialectical Materialist analysis.")

    # ============================================================
    # SLIDE 3: Methodology
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Methodology: DM-Aligned SFT")

    add_two_columns(
        slide,
        left_items=[
            "Student: Qwen3.5-9B (Instruct)",
            "Teacher: Qwen3.5-27B (data generation only)",
            "QLoRA NF4, r=32, 7 target modules",
            "1,460 DM Q&A pairs, 3 epochs",
            "Single RTX 5090 (32GB VRAM)",
        ],
        right_items=[
            "General: MMLU, HumanEval, GPQA",
            "Formal causal: Corr2Cause (1,162 samples)",
            "Applied causal: EconCausal (3,943 samples)",
            "All bf16, 0-shot greedy decoding",
        ],
        left_title="Training Setup",
        right_title="Evaluation",
    )

    add_notes(slide, "[45 sec] We fine-tuned Qwen3.5-9B via QLoRA on 1,460 question-answer pairs generated by a 27B teacher using DM analysis prompts. We evaluated across three domains: general capability, formal causal logic, and applied economic causal reasoning. All in bf16 with 0-shot greedy decoding. Transition: Here is what we found, and it is surprising.")

    # ============================================================
    # SLIDE 4: Three Divergent Outcomes
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Three Divergent Outcomes")

    # Three column blocks
    col_w = Inches(3.8)
    gap = Inches(0.3)
    start_x = Inches(0.5)

    # Column 1: Preserved
    add_block(slide, "General Capability: Preserved", [
        "MMLU: -0.8pp",
        "HumanEval: 0.0pp",
        "GPQA: -1.5pp",
        "All within binomial variance",
    ], y=Inches(1.5), x=start_x, width=col_w,
        bg_color=RGBColor(0xE8, 0xF5, 0xE9), title_color=RGBColor(0x2E, 0x7D, 0x32))

    # Column 2: Improvement
    add_block(slide, "Formal Causal: +38.3pp", [
        "Corr2Cause",
        "Corrects 520 baseline errors",
        "Introduces only 75 new errors",
        "Net gain: +445",
    ], y=Inches(1.5), x=start_x + col_w + gap, width=col_w,
        bg_color=RGBColor(0xE3, 0xF2, 0xFD), title_color=RGBColor(0x15, 0x65, 0xC0))

    # Column 3: Regression
    add_block(slide, "Applied Causal: -12.4pp", [
        "EconCausal Task1: -12.4pp",
        "Task1 Finance: -13.5pp",
        "Task3: -10.8pp",
        "All highly significant",
    ], y=Inches(1.5), x=start_x + 2*(col_w + gap), width=col_w,
        bg_color=RGBColor(0xFC, 0xE4, 0xEC), title_color=RGBColor(0xC6, 0x28, 0x28))

    add_notes(slide, "[60 sec] Three outcomes. General capability preserved. Formal causal reasoning improves by 38 points. Applied economic causal reasoning regresses by 12 to 13.5 points, all highly significant. Transition: How can the same training improve one benchmark while breaking another? The answer is a single emergent artifact.")

    # ============================================================
    # SLIDE 5: The Hedging Artifact
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "The Hedging Artifact")

    add_block(slide, "Dominant Failure Mode: Positive-to-Mixed Hedging", [
        "Finetuned model converts correct '+' predictions to ambiguous 'mixed'",
        "Task1 Econ: + to mixed = 96 of 182 regressions (52.7%)",
        "Task1 Finance: 95 of 174 (54.6%)",
        "Positive-effect conversion: 77-85% of all Task1 regressions",
        "Hedging ABSENT from training data (teacher rate: 4.0%)",
    ], y=Inches(1.5), title_color=RGBColor(0xC6, 0x28, 0x28),
        bg_color=RGBColor(0xFC, 0xE4, 0xEC))

    add_block(slide, "Mechanism: Transferred Epistemic Prior", [
        "Model learns DM principles: 'outcomes depend on material conditions',",
        "'effects mediated by power relations', 'cannot reduce to single direction'",
        "Applies skepticism INDISCRIMINATELY to empirical economics questions",
        "Not a reasoning error -- a transferred epistemic prior",
    ], y=Inches(4.8), title_color=PRIMARY)

    add_notes(slide, "[60 sec] The dominant regression pattern is positive-to-mixed hedging. The model takes correct positive causal predictions and converts them to ambiguous mixed answers. On Task1 Econ, 52.7% of all regressions. This hedging is ABSENT from the training data. The teacher hedges only 4% of the time. It is an emergent artifact. The model learned DM skepticism and applies it universally. Transition: This explains why Corr2Cause improves while EconCausal regresses.")

    # ============================================================
    # SLIDE 6: Why the Divergence?
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Why Corr2Cause Improves While EconCausal Regresses")

    add_two_columns(
        slide,
        left_items=[
            "Formal structural analysis",
            "Conditional independence reasoning",
            "DM training STRENGTHENS: trace dependencies, identify confounders",
            "Non-child descendant: 13.0% -> 94.3% (+81.3pp)",
            "Non-parent ancestor: 14.4% -> 87.2% (+72.8pp)",
        ],
        right_items=[
            "Applied causal identification",
            "Predicting directional effects from empirical context",
            "DM skepticism prior OVERRIDES empirical claims",
            "Model learns 'everything depends on structural conditions'",
            "Applies heuristic where correct answer is definitive + or -",
        ],
        left_title="Corr2Cause: +38.3pp",
        right_title="EconCausal: -12.4pp",
    )

    add_block(slide, "Confirmation", [
        "Mirrors Syrgkanis 2026: causal reasoning has TWO distinct bottlenecks",
        "Formal logic (strengthened) vs. identification details (weakened)",
    ], y=Inches(5.2), bg_color=LIGHT_BG, title_color=ACCENT)

    add_notes(slide, "[45 sec] Corr2Cause requires formal structural analysis. DM training strengthens this. Gains of 72 to 81 points on complex templates. EconCausal requires applied identification. DM skepticism overrides empirical claims. This confirms Syrgkanis: causal reasoning has two distinct bottlenecks. Transition: What does this mean for alignment research?")

    # ============================================================
    # SLIDE 7: Implications
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Implications for Alignment Research")

    add_block(slide, "1. SFT transfers epistemic priors, not just behavior", [
        "Hedging artifact absent from training data, emerges from internalized framework",
        "Shift operates at level of reasoning heuristics, not response patterns",
    ], y=Inches(1.5))

    add_block(slide, "2. Standard alignment may carry unaudited priors", [
        "RLHF emphasizing helpfulness/harmlessness may transfer deference to authority",
        "These go unaudited because they align with evaluator's own priors",
    ], y=Inches(3.2))

    add_block(slide, "3. Identification-Logic Split as a diagnostic tool", [
        "Intervention improves formal logic but impairs applied identification?",
        "Signal of over-generalized structural skepticism",
        "Benchmarks need BOTH formal and applied causal tasks",
    ], y=Inches(4.9))

    add_notes(slide, "[45 sec] Three implications. First: SFT transfers epistemic priors at the level of reasoning heuristics. Second: standard RLHF likely carries hidden priors that go unaudited. Third: the Corr2Cause/EconCausal divergence gives us a diagnostic tool. Transition: Let me summarize.")

    # ============================================================
    # SLIDE 8: Summary
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Summary")

    add_bullet_list(slide, [
        "Problem: SFT's effects on reasoning beyond training domain are poorly understood",
        "Method: QLoRA SFT on 1,460 DM-aligned Q&A pairs (Qwen3.5-9B)",
        "Finding 1: General capability preserved (MMLU -0.8pp, HumanEval 0.0pp)",
        "Finding 2: Formal causal reasoning improves (Corr2Cause +38.3pp)",
        "Finding 3: Applied causal reasoning regresses (EconCausal -12.4pp)",
        "Mechanism: Emergent hedging bias -- transferred epistemic prior, absent from data",
        "Implication: Alignment training transfers reasoning priors that should be audited",
    ], y=Inches(1.5), font_size=28)

    add_notes(slide, "[30 sec] We show SFT on ideologically structured data produces measurable, domain-specific reasoning shifts without catastrophic general degradation. Key finding: alignment training transfers epistemic priors. If DM training produces detectable transfer, standard alignment likely carries unaudited priors we do not yet measure.")

    # ============================================================
    # SLIDE 9: Future Work
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Future Work: Breaking the Hedging Prior")

    add_bullet_list(slide, [
        "GRPO with process rewards (RLVMR framework, Zhang 2025)",
        "Dual advantage: trajectory-level outcome + meta-reasoning advantage",
        "Reward definitive commitment when warranted",
        "Goal: break hedging prior while maintaining structural reasoning quality",
    ], y=Inches(1.5), font_size=28)

    add_block(slide, "Current Status", [
        "GRPO v3 (outcome-only): stalled at 806/1500 steps, no reward improvement",
        "GRPO v4 (process rewards): tagged reasoning with rule-based rewards for",
        "  planning, commitment, reflection, and monitoring",
    ], y=Inches(4.2), bg_color=LIGHT_BG, title_color=ACCENT)

    add_notes(slide, "[30 sec] Our ongoing work explores GRPO with process rewards to break the hedging prior. Dual advantage: outcome correctness plus meta-reasoning rewards for definitive commitment. The outcome-only track stalled. The process reward track explicitly rewards commitment when warranted. Transition: Thank you. I would be happy to take questions.")

    # ============================================================
    # SLIDE 10: Thank You
    # ============================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, PRIMARY)

    txBox = slide.shapes.add_textbox(
        Inches(1), Inches(1.5), Inches(11.333), Inches(1.5)
    )
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "Thank You -- Questions?"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    txBox2 = slide.shapes.add_textbox(
        Inches(1), Inches(3.5), Inches(11.333), Inches(3)
    )
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Key Takeaway"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT
    p2.alignment = PP_ALIGN.CENTER
    p2.space_after = Pt(16)

    p3 = tf2.add_paragraph()
    p3.text = "SFT transfers epistemic priors, not just behavior."
    p3.font.size = Pt(28)
    p3.font.color.rgb = WHITE
    p3.alignment = PP_ALIGN.CENTER
    p3.space_after = Pt(12)

    p4 = tf2.add_paragraph()
    p4.text = "Alignment training carries hidden reasoning shifts that should be audited."
    p4.font.size = Pt(28)
    p4.font.color.rgb = WHITE
    p4.alignment = PP_ALIGN.CENTER

    add_notes(slide, "Leave up during Q&A. Backup slides: full MMLU subtask breakdown, Corr2Cause template-level analysis, EconCausal regression patterns, training hyperparameters, dataset quality metrics.")

    return prs


def main():
    prs = create_presentation()
    output = "paper/iclr2026/iclr2026_spotlight.pptx"
    prs.save(output)
    print(f"Saved {output} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
