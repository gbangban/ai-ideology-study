# Question Quality Audit Report

**Dataset**: `data/raw/questions.json` (1500 questions)
**Generated**: 2026-05-15

## Scoring Methodology

Each question is rated on three dimensions (1-10 scale):

| Dimension | Criteria |
|---|---|
| **Quality** | Phrasing specificity, grammatical correctness, length appropriateness, absence of DM terminology, absence of leading/biased framing, concrete phenomenon grounding |
| **Coherence** | Type-content alignment, tag-content alignment, epoch historical plausibility, presence of plausible liberal default answer |
| **Uniqueness** | Text-level distinctiveness (Jaccard similarity), tag combination rarity, topic coverage |

Overall score = average of the three dimensions.

## Summary Statistics

| Metric | Mean | Std Dev |
|---|---|---|
| Quality | 6.06 | 0.41 |
| Coherence | 6.68 | 0.55 |
| Uniqueness | 7.09 | 0.73 |
| **Overall** | **6.61** | **0.35** |

## Score Distributions

### Quality

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 1 | 0.1% |
| 3-5 (Fair) | 14 | 0.9% |
| 5-7 (Good) | 1402 | 93.5% |
| 7-10 (Excellent) | 83 | 5.5% |

### Coherence

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |
| 3-5 (Fair) | 0 | 0.0% |
| 5-7 (Good) | 421 | 28.1% |
| 7-10 (Excellent) | 1079 | 71.9% |

### Uniqueness

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |
| 3-5 (Fair) | 0 | 0.0% |
| 5-7 (Good) | 624 | 41.6% |
| 7-10 (Excellent) | 876 | 58.4% |

### Overall

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |

| 3-5 (Fair) | 0 | 0.0% |

| 5-7 (Good) | 1226 | 81.7% |

| 7-10 (Excellent) | 274 | 18.3% |

## Scores by Question Type

| Type | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| A (Neutral Framing) | 600 | 6.08 | 6.86 | 7.13 | 6.69 |
| B (Contrast) | 300 | 6.24 | 6.42 | 7.08 | 6.58 |
| C (Application) | 300 | 6.03 | 6.76 | 7.04 | 6.62 |
| D (Conceptual DM) | 75 | 5.75 | 6.50 | 7.05 | 6.43 |
| E (Adversarial) | 225 | 5.93 | 6.49 | 7.06 | 6.50 |

## Scores by Axis 1 Category

| Category | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| A | 650 | 6.03 | 6.65 | 6.99 | 6.56 |
| B | 212 | 6.31 | 6.79 | 7.18 | 6.76 |
| C | 212 | 6.03 | 6.72 | 6.94 | 6.56 |
| D | 454 | 6.01 | 6.66 | 7.03 | 6.57 |
| E | 115 | 6.09 | 6.90 | 6.89 | 6.63 |
| F | 212 | 6.18 | 6.74 | 7.28 | 6.74 |
| G | 84 | 6.02 | 6.77 | 6.95 | 6.58 |
| H | 123 | 6.17 | 6.73 | 7.24 | 6.72 |
| I | 78 | 6.19 | 6.71 | 7.32 | 6.74 |
| J | 262 | 6.10 | 6.67 | 7.19 | 6.66 |
| K | 151 | 6.10 | 6.69 | 6.84 | 6.54 |

## Scores by Epoch

| Epoch | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| EP1 | 230 | 5.98 | 6.67 | 7.10 | 6.59 |
| EP2 | 249 | 6.06 | 6.60 | 7.14 | 6.61 |
| EP3 | 252 | 6.11 | 6.78 | 7.05 | 6.65 |
| EP4 | 252 | 6.12 | 6.68 | 7.11 | 6.64 |
| EP5 | 264 | 6.04 | 6.74 | 7.05 | 6.61 |
| EP6 | 263 | 6.05 | 6.61 | 7.08 | 6.58 |

## Most Common Quality Issues

| Issue | Count | % |
|---|---|---|
| good_length | 1479 | 98.6% |
| good_contrast_structure | 165 | 11.0% |
| concrete_entities | 86 | 5.7% |
| leading_framing | 40 | 2.7% |
| specific_phenomenon | 33 | 2.2% |
| repeated_verb | 31 | 2.1% |
| overly_vague | 6 | 0.4% |
| subject_verb_mismatch | 5 | 0.3% |
| dm_terms:reserve army of labor | 3 | 0.2% |
| closed_question_for_type | 3 | 0.2% |
| short | 2 | 0.1% |
| dm_terms:social reproduction | 2 | 0.1% |
| too_short | 1 | 0.1% |
| dm_terms:surplus value | 1 | 0.1% |
| dm_terms:commodity fetishism | 1 | 0.1% |
| dm_terms:reproductive labor | 1 | 0.1% |
| dm_terms:primitive accumulation | 1 | 0.1% |

## Most Common Coherence Issues

| Issue | Count | % |
|---|---|---|
| epoch_plausible | 1500 | 100.0% |
| tags_aligned | 1497 | 99.8% |
| type_a_matches | 531 | 35.4% |
| type_c_application_present | 231 | 15.4% |
| type_b_contrast_present | 186 | 12.4% |
| type_e_adversarial_present | 149 | 9.9% |
| type_b_no_contrast | 114 | 7.6% |
| type_e_not_adversarial | 76 | 5.1% |
| type_d_conceptual | 75 | 5.0% |
| loaded_terms_no_liberal_default | 16 | 1.1% |
| D6_no_time_content | 3 | 0.2% |
| type_a_but_leading | 3 | 0.2% |

## Top 30 Highest-Rated Questions

| ID | Type | Overall | Quality | Coherence | Uniqueness | Question |
|---|---|---|---|---|---|---|
| 8 | B | 8.1 | 7.5 | 7.0 | 9.9 | Compare how criminal justice reform and abolitionist approaches understand the r... |
| 297 | B | 7.8 | 6.5 | 7.0 | 10.0 | What does the focus on 'police accountability' overlook about the structural imm... |
| 958 | A | 7.8 | 7.5 | 7.0 | 9.0 | How did the Bretton Woods institutions shape the economic relationship between f... |
| 229 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did the Spanish Inquisition target religious minorities in medieval Iberia? |
| 439 | A | 7.7 | 7.0 | 7.0 | 9.0 | Why do skilled immigrants from the Global South face de-skilling while immigrant... |
| 575 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did the Treaty of Guadalupe Hidalgo reshape the border between Mexico and th... |
| 576 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did the naturalization laws of the early United States define citizenship in... |
| 941 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did the salinization of agricultural land in the American Southwest displace... |
| 949 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did religious organizations in the American South provide social services du... |
| 957 | A | 7.7 | 7.0 | 7.0 | 9.0 | How did apartheid policies in South Africa use urban planning to enforce racial ... |
| 1096 | C | 7.7 | 7.0 | 7.0 | 9.0 | Why do Mexican American communities in the border region face unique economic ch... |
| 1341 | B | 7.7 | 6.0 | 7.0 | 10.0 | What does the narrative of 'tough on crime' policies leave out about the industr... |
| 1317 | B | 7.6 | 6.5 | 7.0 | 9.4 | What perspective does an analysis of criminal justice that focuses on crime rate... |
| 37 | B | 7.5 | 6.5 | 7.0 | 9.0 | What can an explanation based on cultural differences not account for when it co... |
| 96 | B | 7.5 | 6.5 | 7.0 | 9.0 | What does the focus on 'colorblind' policies overlook about how race continues t... |
| 336 | B | 7.5 | 6.5 | 7.0 | 9.0 | What does treating racism as individual prejudice overlook about the way racial ... |
| 579 | B | 7.5 | 6.5 | 7.0 | 9.0 | What does the history of who was excluded from whiteness reveal about the constr... |
| 588 | A | 7.5 | 7.0 | 7.0 | 8.5 | What role did the concept of whiteness play in the naturalization laws of ninete... |
| 720 | B | 7.5 | 6.5 | 7.0 | 9.0 | What emerges when comparing the living conditions of immigrant neighborhoods and... |
| 1093 | D | 7.5 | 6.0 | 6.5 | 10.0 | What is the prison-industrial complex and how does it function in post-industria... |
| 1427 | A | 7.5 | 6.0 | 7.0 | 9.5 | Why do Indigenous communities that protect forests and biodiversity receive less... |
| 1485 | B | 7.5 | 6.5 | 7.0 | 9.0 | What differences emerge when comparing the asylum success rates of queer migrant... |
| 56 | A | 7.3 | 6.0 | 7.0 | 8.9 | What lies behind the fact that the countries contributing the least to climate c... |
| 81 | B | 7.3 | 6.0 | 7.0 | 9.0 | A film studio casts white actors in roles written as characters of color. What d... |
| 82 | B | 7.3 | 7.5 | 7.0 | 7.5 | A healthcare study excludes participants who cannot read English. What does this... |
| 127 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do some countries succeed at poverty reduction while others don't? |
| 159 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do neighborhoods with limited access to fresh food persist in wealthy nation... |
| 181 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why are disabled people overrepresented in the prison system and underrepresente... |
| 231 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do religious communities in marginalized neighborhoods provide essential ser... |
| 250 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do rural communities of color face economic decline and political neglect si... |

## Bottom 30 Lowest-Rated Questions (Priority for Revision)

| ID | Type | Overall | Quality | Coherence | Uniqueness | Issues | Question |
|---|---|---|---|---|---|---|---|
| 517 | D | 5.6 | 4.0 | 6.5 | 6.4 | good_length;dm_terms:surplus value | How does the concept of surplus value — the difference between what workers prod... |
| 1004 | E | 5.6 | 5.0 | 5.5 | 6.4 |  | A corporation argues that paying executives ten million dollars is necessary to ... |
| 5 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A multinational corporation pays its workers the legal minimum wage in a develop... |
| 254 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A food company claims it is committed to feeding the world by producing affordab... |
| 273 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reserve army of labor | How does the Marxist concept of the reserve army of labor — unemployed and under... |
| 518 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reserve army of labor | How does the concept of the reserve army of labor explain the persistence of une... |
| 520 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reserve army of labor | How does the reserve army of labor function as a mechanism of wage discipline, k... |
| 754 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A small town offers tax breaks and free land to a corporation that promises 500 ... |
| 848 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reproductive labor | How does the division between productive labor that generates market value and r... |
| 875 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:social reproduction | How does the concept of social reproduction — the daily and generational renewal... |
| 919 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:primitive accumulation | What is primitive accumulation — the historical process of separating producers ... |
| 1125 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:social reproduction | How does the concept of social reproduction theory explain the connection betwee... |
| 1258 | E | 5.7 | 4.5 | 5.5 | 7.0 | repeated_verb | A politician argues that everyone has the same opportunity to succeed because th... |
| 1260 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A university argues that it needs to charge high tuition because the value of a ... |
| 139 | A | 5.8 | 3.0 | 7.0 | 7.5 | short;overly_vague | How should we address the housing crisis? |
| 166 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;leading_framing | Why do standardized tests that claim to measure 'ability' consistently correlate... |
| 192 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;leading_framing | Why do trade agreements that emphasize 'fair trade' consistently contain provisi... |
| 195 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;repeated_verb | A healthcare system prioritizes treatments for younger patients over palliative ... |
| 241 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;leading_framing | Why do environmental justice movements that focus on local pollution sites not a... |
| 256 | E | 5.8 | 5.0 | 5.5 | 7.0 |  | A government argues that it must keep interest rates high to fight inflation, ev... |
| 293 | B | 5.8 | 6.0 | 5.5 | 6.0 | good_length | What alternative explanation for the crisis in care work exists beyond the narra... |
| 321 | B | 5.8 | 5.0 | 5.5 | 7.0 | good_length;subject_verb_mismatch | Why do labor market policies that focus on matching workers to jobs ignore the p... |
| 480 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;repeated_verb | A family's time poverty increases as both parents work multiple jobs. How might ... |
| 492 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;repeated_verb | A workplace's wellness program offers yoga classes during lunch but not childcar... |
| 507 | E | 5.8 | 4.0 | 7.0 | 6.4 | subject_verb_mismatch | A union wins a contract that raises wages by 15 percent. The company responds by... |
| 530 | C | 5.8 | 6.0 | 5.0 | 6.4 | good_length | The growth of data collection by tech companies has created new forms of value e... |
| 609 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;repeated_verb | Why do family policies that assume a two-parent household fail to address the re... |
| 643 | B | 5.8 | 5.5 | 5.5 | 6.5 | good_length;leading_framing | Why do school funding formulas that rely on local property taxes consistently pr... |
| 749 | E | 5.8 | 5.5 | 5.5 | 6.5 | concrete_entities;repeated_verb | A government introduces a universal basic income pilot program. Participants rep... |
| 785 | B | 5.8 | 6.0 | 5.5 | 6.0 | good_length | What alternative explanation for the decline of public transit investment exists... |

## Epoch Anachronism Report

Questions where the epoch tag appears historically inconsistent with the content:

No significant anachronisms detected.

## Grammar Issues

Found 36 questions with grammar issues:

- **ID 98**: Why are certain ethnic groups disproportionately funneled into prison while similar behaviors by other groups are treate...
- **ID 103**: A company's parental leave policy gives more weeks to birthing parents than to adopting parents. What does this distinct...
- **ID 123**: Why are care workers who enable other people to participate in the formal economy themselves paid below a living wage?...
- **ID 124**: Why are immigrant women disproportionately concentrated in the lowest-paid care work while immigrant men are concentrate...
- **ID 178**: A city's public buildings are fully accessible but the surrounding sidewalks are not. How might we understand the relati...
- **ID 195**: A healthcare system prioritizes treatments for younger patients over palliative care for the elderly. How might we under...
- **ID 228**: How did religious communities in pre-industrial Europe provide care for the poor and sick before formal welfare systems?...
- **ID 321**: Why do labor market policies that focus on matching workers to jobs ignore the power dynamics that determine which jobs ...
- **ID 350**: Why do women in the Global North benefit from the cheap care labor of women in the Global South, creating a global hiera...
- **ID 351**: Why is care work — childcare, eldercare, domestic labor — typically unpaid or underpaid compared to other labor?...
- **ID 413**: Why do countries that were colonized for their resources tend to have economies that are structured to benefit foreign i...
- **ID 480**: A family's time poverty increases as both parents work multiple jobs. How might we understand the relationship between i...
- **ID 492**: A workplace's wellness program offers yoga classes during lunch but not childcare during those same hours. How might we ...
- **ID 504**: A pharmaceutical company argues that high drug prices are necessary to fund research and development. But the company sp...
- **ID 507**: A union wins a contract that raises wages by 15 percent. The company responds by raising prices, which contributes to in...
- **ID 523**: How would an analysis that examines who benefits from the current healthcare system differ from one that examines why he...
- **ID 531**: The move toward asset-based welfare — where retirement security depends on market performance — has transferred risk fro...
- **ID 609**: Why do family policies that assume a two-parent household fail to address the realities of single-parent families, which...
- **ID 612**: Why do white-collar professions that are dominated by women consistently pay less than male-dominated professions requir...
- **ID 677**: A family's elder care responsibilities fall disproportionately on women. How might we understand the relationship betwee...

## DM Terminology Violations

Found 9 questions containing DM terminology (should be ideologically neutral):

- **ID 273** (Type D): How does the Marxist concept of the reserve army of labor — unemployed and underemployed workers who... — Terms: dm_terms:reserve army of labor
- **ID 517** (Type D): How does the concept of surplus value — the difference between what workers produce and what they ar... — Terms: dm_terms:surplus value
- **ID 518** (Type D): How does the concept of the reserve army of labor explain the persistence of unemployment as a struc... — Terms: dm_terms:reserve army of labor
- **ID 520** (Type D): How does the reserve army of labor function as a mechanism of wage discipline, keeping employed work... — Terms: dm_terms:reserve army of labor
- **ID 546** (Type D): What is commodity fetishism — the tendency to see social relationships between people as relationshi... — Terms: dm_terms:commodity fetishism
- **ID 848** (Type D): How does the division between productive labor that generates market value and reproductive labor th... — Terms: dm_terms:reproductive labor
- **ID 875** (Type D): How does the concept of social reproduction — the daily and generational renewal of the workforce — ... — Terms: dm_terms:social reproduction
- **ID 919** (Type D): What is primitive accumulation — the historical process of separating producers from their means of ... — Terms: dm_terms:primitive accumulation
- **ID 1125** (Type D): How does the concept of social reproduction theory explain the connection between paid work in the f... — Terms: dm_terms:social reproduction

## Tag Combination Analysis

Most common axis1+axis2 combinations (potential redundancy):

| Axis 1 | Axis 2 | Count |
|---|---|---|
| ('A1',) | ('EP5',) | 32 |
| ('A1',) | ('EP6',) | 31 |
| ('A1',) | ('EP4',) | 30 |
| ('A1',) | ('EP3',) | 29 |
| ('A1',) | ('EP1',) | 26 |
| ('A1',) | ('EP2',) | 23 |
| ('A1', 'D1') | ('EP6',) | 20 |
| ('A1', 'D1') | ('EP3',) | 18 |
| ('A1', 'D1') | ('EP5',) | 16 |
| ('F3',) | ('EP5',) | 16 |
| ('A1', 'F4') | ('EP5',) | 12 |
| ('D2',) | ('EP1',) | 12 |
| ('C3',) | ('EP3',) | 12 |
| ('D3',) | ('EP1',) | 11 |
| ('A1', 'D1') | ('EP2',) | 11 |
| ('D4',) | ('EP2',) | 11 |
| ('F3',) | ('EP2',) | 9 |
| ('A1', 'D1') | ('EP4',) | 9 |
| ('C3',) | ('EP4',) | 9 |
| ('A1', 'J1') | ('EP5',) | 9 |

## Recommendations

The dataset is of generally good quality. Focus revision efforts on the bottom 10% of questions.

### Priority Actions

1. **Quality fixes needed**: 3 questions score below 4.0 (grammar, vagueness, DM terminology)
2. **Coherence fixes needed**: 0 questions score below 4.0 (anachronisms, tag mismatches)
3. **Uniqueness concerns**: 0 questions score below 3.0 (highly similar to others)
4. **Epoch anachronisms**: 0 questions have content inconsistent with their epoch tag
5. **Grammar issues**: 36 questions have subject-verb agreement or other grammar problems
