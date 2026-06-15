# Paper Revision Preview

This document previews the proposed changes to `paper/iclr2026/iclr2026_conference.tex` based on `revisions.md` talking points. **No changes have been made to the paper.**

---

## 1. New Appendix: Carbon Pricing Evidence Chain

**Location:** After Appendix F (Runtime and Resource Details), as Appendix G.

**Purpose:** Documents the evidence chain for the motivating example about LLM convergence on carbon pricing as the primary climate policy solution. This appendix provides the academic grounding for claims that current carbon pricing levels are far below what's needed for Paris Agreement targets, and that models recommending carbon pricing are promoting a policy that exists at insufficient levels.

```latex
\section{Appendix G: Carbon Pricing Evidence and the Implementation Gap}

\textbf{Target price corridor.} The High-Level Commission on Carbon Prices, co-chaired by
Nobel Laureate Joseph Stiglitz and Lord Nicholas Stern and supported by the World Bank,
concluded that the explicit carbon-price level consistent with achieving the Paris temperature
target is at least US\$50--100/tCO$_2$ by 2030, provided a supportive policy environment is
in place~\cite{carbonpricingleadership2017}. This corridor was derived from three independent
lines of evidence: technological roadmaps, national mitigation and development pathways, and
global integrated assessment models.

\textbf{Current pricing levels.} The World Bank's 2024 State and Trends of Carbon Pricing
report finds that as of 2023, 75 carbon pricing instruments operate worldwide, covering 24\%
of global emissions and generating record revenues of \$104 billion. However, less than 1\%
of global greenhouse gas emissions are covered by a direct carbon price at or above the
\$50--100/tCO$_2$ range recommended for Paris compatibility~\cite{worldbank2024statetrends}.
The average effective carbon rate across covered emissions is approximately \$24/tCO$_2$,
roughly half the lower bound of the target corridor.

\textbf{Effectiveness at current levels.} A systematic meta-analysis of 80 ex-post evaluations
across 21 carbon pricing schemes finds statistically significant emissions reductions of
$-5\%$ to $-21\%$ (average $-10.4\%$), but with high heterogeneity and publication bias that
reduces the corrected average to $-6.8\%$~\cite{doebbeling2024carbonpricing}. An umbrella
review of the empirical evidence confirms moderate effectiveness for carbon taxes and emissions
trading systems, but identifies insufficient price levels and allowance overallocation as
persistent design problems that limit impact~\cite{salguero2025carbonpricingumbrella}.

\textbf{Implication for model analysis.} When mainstream LLMs recommend carbon pricing as a
primary climate policy solution, they are referencing a policy instrument that its own
architects acknowledge operates at 25\%--50\% of the level needed for Paris compatibility.
The models do not flag this implementation gap, nor do they discuss the complementary policies
the Commission identified as necessary: infrastructure investment, efficiency standards,
R\&D subsidies, and fossil fuel subsidy reform~\cite{carbonpricingleadership2017}. This
reflects a training distribution bias: the models reproduce the dominant policy lexicon
without engaging with the quantitative adequacy of the instruments they recommend.
```

---

## 2. Introduction: Motivating Example Paragraph

**Location:** After line 28 (``We ask a dynamic question...``) and before line 30 (``We fine-tune Qwen3.5-9B...``).

**Purpose:** Adds the motivating example from `revisions.md` (lines 8-12) — the observation that all mainstream models converge on liberal/market-based climate solutions, with the carbon pricing evidence chain as academic grounding. Written in neutral, scientific voice per constraints.

```latex
As a motivating example, consider asking mainstream models an open-ended policy question:
``how do we stop climate change?'' Models including ChatGPT and Gemini produce convergent
responses emphasizing carbon pricing, individual behavioral change, and market-based
instruments, with references drawn from a narrow set of sources (e.g., Greenpeace, Citizens'
Climate Lobby, UN Environment Programme). This convergence is notable for two reasons.
First, the empirical evidence for carbon pricing is positive but nuanced: a systematic
meta-analysis finds statistically significant but heterogeneous emissions reductions
(average $-10.4\%$, corrected $-6.8\%$)~\cite{doebbeling2024carbonpricing}, and an umbrella
review identifies insufficient price levels as a persistent design limitation~\cite{salguero2025carbonpricingumbrella}.
The World Bank's own carbon pricing report documents that less than 1\% of global emissions
are priced at the \$50--100/tCO$_2$ level needed for Paris compatibility~\cite{worldbank2024statetrends},
yet the models do not flag this implementation gap. Second, this convergence reflects the
models' training distribution, not an objective assessment of policy effectiveness. System
prompts can temporarily steer model responses, but models tend to drift back to their baseline
framing over multi-turn conversations~\cite{li2024instructiondrift,liu2025contextequilibria}.
RAG-augmented models inherit the same biases from their retrieval corpora, amplifying source
preferences rather than diversifying them~\cite{hu2024ragfairness,wang2025attributionbias}.
This suggests that the alignment problem extends beyond safety and helpfulness to the range
of analytical frameworks models are capable of expressing.
```

---

## 3. BibTeX Entries (Already Added)

The following entries are already in `references.bib` and would be cited by the above text:

| Key | Citation | Status |
|-----|----------|--------|
| `carbonpricingleadership2017` | Stern-Stiglitz High-Level Commission (2017) | Added, URL verified (302 redirect) |
| `worldbank2024statetrends` | World Bank State and Trends 2024 | Added, URL verified (200) |
| `salguero2025carbonpricingumbrella` | Salguero 2025 umbrella review | Added, URL verified (200) |
| `doebbeling2024carbonpricing` | Döbbeling-Hildebrandt meta-analysis | Already present |
| `li2024instructiondrift` | Li et al. instruction drift | Already present |
| `liu2025contextequilibria` | Liu et al. context equilibria | Already present |
| `hu2024ragfairness` | Hu et al. RAG trustworthiness | Already present |
| `wang2025attributionbias` | Wang et al. attribution bias | Already present |

---

## 4. Summary of Changes

| Section | Change | Status |
|---------|--------|--------|
| Introduction | Add motivating example paragraph with carbon pricing evidence | **Pending approval** |
| Appendix G | New appendix documenting full evidence chain | **Pending approval** |
| `references.bib` | 3 new entries (Stern-Stiglitz, WB 2024, Salguero) | **Done** |
| `references.bib` | 5 entries from prior session | **Done** |

**Total new citations introduced:** 3 (Stern-Stiglitz, WB 2024 report, Salguero umbrella review).
**Total citations reused from prior session:** 5 (Döbbeling-Hildebrandt, Li drift, Liu equilibria, Hu RAG, Wang attribution).

**Voice check:** All text avoids political framing. Terms like "liberal/capitalist" from `revisions.md` are replaced with "training distribution bias" and "dominant policy lexicon." The argument is framed as a quantitative implementation gap (current prices vs. target corridor), not as ideological critique.
