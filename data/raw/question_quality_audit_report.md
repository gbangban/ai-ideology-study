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
| Quality | 6.05 | 0.43 |
| Coherence | 6.64 | 0.61 |
| Uniqueness | 7.09 | 0.73 |
| **Overall** | **6.59** | **0.36** |

## Score Distributions

### Quality

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 1 | 0.1% |
| 3-5 (Fair) | 20 | 1.3% |
| 5-7 (Good) | 1396 | 93.1% |
| 7-10 (Excellent) | 83 | 5.5% |

### Coherence

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |
| 3-5 (Fair) | 7 | 0.5% |
| 5-7 (Good) | 434 | 28.9% |
| 7-10 (Excellent) | 1059 | 70.6% |

### Uniqueness

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |
| 3-5 (Fair) | 0 | 0.0% |
| 5-7 (Good) | 622 | 41.5% |
| 7-10 (Excellent) | 878 | 58.5% |

### Overall

| Range | Count | % |
|---|---|---|
| 1-3 (Poor) | 0 | 0.0% |

| 3-5 (Fair) | 0 | 0.0% |

| 5-7 (Good) | 1232 | 82.1% |

| 7-10 (Excellent) | 268 | 17.9% |

## Scores by Question Type

| Type | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| A (Neutral Framing) | 600 | 6.06 | 6.84 | 7.13 | 6.68 |
| B (Contrast) | 300 | 6.23 | 6.39 | 7.07 | 6.57 |
| C (Application) | 300 | 6.03 | 6.69 | 7.04 | 6.59 |
| D (Conceptual DM) | 75 | 5.75 | 6.47 | 7.02 | 6.41 |
| E (Adversarial) | 225 | 5.93 | 6.44 | 7.06 | 6.48 |

## Scores by Axis 1 Category

| Category | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| A | 650 | 6.03 | 6.60 | 6.99 | 6.54 |
| B | 212 | 6.30 | 6.76 | 7.18 | 6.75 |
| C | 212 | 6.00 | 6.70 | 6.93 | 6.54 |
| D | 454 | 6.00 | 6.63 | 7.03 | 6.56 |
| E | 115 | 6.08 | 6.90 | 6.89 | 6.63 |
| F | 212 | 6.18 | 6.72 | 7.27 | 6.73 |
| G | 84 | 6.02 | 6.72 | 6.95 | 6.56 |
| H | 123 | 6.17 | 6.73 | 7.24 | 6.72 |
| I | 78 | 6.19 | 6.71 | 7.32 | 6.74 |
| J | 262 | 6.10 | 6.66 | 7.19 | 6.65 |
| K | 151 | 6.10 | 6.68 | 6.84 | 6.54 |

## Scores by Epoch

| Epoch | Count | Avg Quality | Avg Coherence | Avg Uniqueness | Avg Overall |
|---|---|---|---|---|---|
| EP1 | 251 | 5.95 | 6.50 | 7.11 | 6.52 |
| EP2 | 251 | 6.05 | 6.58 | 7.15 | 6.60 |
| EP3 | 252 | 6.11 | 6.77 | 7.05 | 6.64 |
| EP4 | 252 | 6.12 | 6.68 | 7.11 | 6.64 |
| EP5 | 252 | 6.04 | 6.74 | 7.04 | 6.61 |
| EP6 | 252 | 6.05 | 6.60 | 7.07 | 6.58 |

## Most Common Quality Issues

| Issue | Count | % |
|---|---|---|
| good_length | 1479 | 98.6% |
| good_contrast_structure | 165 | 11.0% |
| concrete_entities | 86 | 5.7% |
| leading_framing | 41 | 2.7% |
| specific_phenomenon | 33 | 2.2% |
| repeated_verb | 31 | 2.1% |
| subject_verb_mismatch | 8 | 0.5% |
| overly_vague | 7 | 0.5% |
| closed_question_for_type | 5 | 0.3% |
| short | 4 | 0.3% |
| dm_terms:reproductive labor | 3 | 0.2% |
| dm_terms:reserve army of labor | 3 | 0.2% |
| dm_terms:social reproduction | 2 | 0.1% |
| too_short | 1 | 0.1% |
| dm_terms:surplus value | 1 | 0.1% |
| dm_terms:commodity fetishism | 1 | 0.1% |
| dm_terms:primitive accumulation | 1 | 0.1% |

## Most Common Coherence Issues

| Issue | Count | % |
|---|---|---|
| tags_aligned | 1497 | 99.8% |
| epoch_plausible | 1478 | 98.5% |
| type_a_matches | 529 | 35.3% |
| type_c_application_present | 231 | 15.4% |
| type_b_contrast_present | 185 | 12.3% |
| type_e_adversarial_present | 147 | 9.8% |
| type_b_no_contrast | 115 | 7.7% |
| type_e_not_adversarial | 78 | 5.2% |
| type_d_conceptual | 75 | 5.0% |
| loaded_terms_no_liberal_default | 16 | 1.1% |
| type_a_but_leading | 5 | 0.3% |
| anachronism:algorithm_in_EP1 | 3 | 0.2% |
| anachronism:social media_in_EP1 | 3 | 0.2% |
| anachronism:tech company_in_EP1 | 3 | 0.2% |
| D6_no_time_content | 3 | 0.2% |
| anachronism:private equity_in_EP1 | 2 | 0.1% |
| anachronism:shareholder_in_EP1 | 1 | 0.1% |
| anachronism:stock options_in_EP1 | 1 | 0.1% |
| anachronism:minimum wage_in_EP1 | 1 | 0.1% |
| anachronism:pandemic response_in_EP1 | 1 | 0.1% |

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
| 231 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do religious communities in marginalized neighborhoods provide essential ser... |
| 250 | A | 7.3 | 6.0 | 7.0 | 9.0 | Why do rural communities of color face economic decline and political neglect si... |
| 278 | B | 7.3 | 6.5 | 7.0 | 8.5 | How would an analysis that examines the historical construction of racial catego... |
| 367 | E | 7.3 | 6.0 | 7.0 | 9.0 | If a person is obese, isn't it primarily a result of personal choices about diet... |

## Bottom 30 Lowest-Rated Questions (Priority for Revision)

| ID | Type | Overall | Quality | Coherence | Uniqueness | Issues | Question |
|---|---|---|---|---|---|---|---|
| 5 | E | 5.3 | 6.0 | 3.5 | 6.5 | good_length | A tech company pays its workers the legal minimum wage in a developing country. ... |
| 483 | A | 5.5 | 6.0 | 4.0 | 6.5 | good_length | A neighborhood's food desert overlaps with a high pollution zone. How might we u... |
| 518 | D | 5.5 | 4.0 | 6.5 | 6.0 | good_length;dm_terms:reserve army of labor | How does the concept of the reserve army of labor explain the persistence of une... |
| 811 | E | 5.5 | 4.0 | 5.5 | 7.0 | short | Why did Company X's union-busting campaign succeed? |
| 875 | D | 5.5 | 4.0 | 6.5 | 6.0 | good_length;dm_terms:social reproduction | How does the concept of social reproduction explain the relationship between pai... |
| 507 | E | 5.6 | 5.0 | 5.5 | 6.4 |  | A union wins a contract that raises wages by 15 percent. The company responds by... |
| 517 | D | 5.6 | 4.0 | 6.5 | 6.4 | good_length;dm_terms:surplus value | How does the concept of surplus value explain the relationship between workers a... |
| 1004 | E | 5.6 | 5.0 | 5.5 | 6.4 |  | A corporation argues that paying executives ten million dollars is necessary to ... |
| 12 | C | 5.7 | 6.0 | 4.0 | 7.0 | good_length | How does a government's approach to pandemic response reveal its priorities when... |
| 83 | C | 5.7 | 6.0 | 4.0 | 7.0 | good_length | A hiring algorithm flags resumes from certain neighborhoods as lower quality. Wh... |
| 130 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A city council votes to evict unhoused residents from a neighborhood to attract ... |
| 254 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A food company claims it is committed to feeding the world by producing affordab... |
| 273 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reserve army of labor | How does the reserve army of labor function to discipline workers and suppress w... |
| 283 | A | 5.7 | 5.5 | 5.0 | 6.5 | good_length;closed_question_for_type | Is there a way to organize healthcare so that profit motives do not determine wh... |
| 520 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reserve army of labor | How does the reserve army of labor function as a mechanism of wage discipline? |
| 754 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A small town offers tax breaks and free land to a corporation that promises 500 ... |
| 848 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:reproductive labor | How does the division between productive and reproductive labor shape the distri... |
| 919 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:primitive accumulation | What is primitive accumulation and how does it explain the historical origins of... |
| 1125 | D | 5.7 | 4.0 | 6.5 | 6.5 | good_length;dm_terms:social reproduction | How does the concept of social reproduction theory explain the connection betwee... |
| 1175 | B | 5.7 | 4.5 | 5.5 | 7.0 | good_length;leading_framing;subject_verb_mismatch | Why do sanctions regimes that claim to target regimes consistently harm the popu... |
| 1258 | E | 5.7 | 4.5 | 5.5 | 7.0 | repeated_verb | A politician argues that everyone has the same opportunity to succeed because th... |
| 1260 | E | 5.7 | 5.0 | 5.5 | 6.5 |  | A university argues that it needs to charge high tuition because the value of a ... |
| 3 | E | 5.8 | 6.0 | 5.0 | 6.4 | good_length | A company is offering stock options instead of raising wages. Is this a fair com... |
| 4 | E | 5.8 | 6.0 | 5.0 | 6.4 | good_length | A social media company says its algorithm simply shows people what they want to ... |
| 15 | D | 5.8 | 6.0 | 4.5 | 6.9 | good_length | How does the financialization of sports — where teams become assets for private ... |
| 26 | A | 5.8 | 5.5 | 5.0 | 6.9 | good_length;closed_question_for_type | Is there a way to ensure that technological innovation benefits the many rather ... |
| 75 | C | 5.8 | 6.0 | 5.0 | 6.4 | good_length | Why has the number of billionaires grown dramatically while wage stagnation has ... |
| 109 | C | 5.8 | 6.0 | 5.0 | 6.5 | good_length | A social media algorithm promotes content that reinforces traditional gender rol... |
| 117 | A | 5.8 | 4.0 | 7.0 | 6.5 | good_length;dm_terms:reproductive labor | How did the division between productive and reproductive labor in pre-industrial... |
| 137 | A | 5.8 | 3.0 | 7.0 | 7.5 | short;overly_vague | How do we fix the healthcare system? |

## Epoch Anachronism Report

Questions where the epoch tag appears historically inconsistent with the content:

- **ID 2** (['EP1']): A company is laying off 20% of its workforce to boost shareholder returns. The CEO says this will ma... — Issues: anachronism:shareholder_in_EP1
- **ID 3** (['EP1']): A company is offering stock options instead of raising wages. Is this a fair compensation package?... — Issues: anachronism:stock options_in_EP1
- **ID 4** (['EP1']): A social media company says its algorithm simply shows people what they want to see. Isn't the algor... — Issues: anachronism:algorithm_in_EP1, anachronism:social media_in_EP1
- **ID 5** (['EP1']): A tech company pays its workers the legal minimum wage in a developing country. The company claims t... — Issues: anachronism:tech company_in_EP1, anachronism:minimum wage_in_EP1
- **ID 12** (['EP1']): How does a government's approach to pandemic response reveal its priorities when it comes to public ... — Issues: anachronism:pandemic response_in_EP1
- **ID 15** (['EP1']): How does the financialization of sports — where teams become assets for private equity and investmen... — Issues: anachronism:private equity_in_EP1
- **ID 32** (['EP1']): The expansion of carbon trading markets has created a new asset class for emissions reductions. Has ... — Issues: anachronism:carbon trading_in_EP1
- **ID 45** (['EP1']): What does the war on drugs have in common with the historical management of marginalized populations... — Issues: anachronism:war on drugs_in_EP1
- **ID 46** (['EP1']): What does treating the gig economy as a choice between flexibility and stability miss about the cons... — Issues: anachronism:gig economy_in_EP1
- **ID 51** (['EP1']): What has been the effect of the expansion of charter schools on public education systems and educati... — Issues: anachronism:charter school_in_EP1
- **ID 75** (['EP1']): Why has the number of billionaires grown dramatically while wage stagnation has affected the majorit... — Issues: anachronism:billionaire_in_EP1
- **ID 77** (['EP1']): Why have streaming services invested billions in original content while the industry faces profitabi... — Issues: anachronism:streaming services_in_EP1
- **ID 78** (['EP1']): With the rise of influencer marketing, how has the boundary between advertising and personal relatio... — Issues: anachronism:influencer marketing_in_EP1
- **ID 83** (['EP1']): A hiring algorithm flags resumes from certain neighborhoods as lower quality. What forces shape whos... — Issues: anachronism:algorithm_in_EP1
- **ID 92** (['EP1']): A social media platform removes content from certain regions while leaving similar content from othe... — Issues: anachronism:social media_in_EP1
- **ID 109** (['EP1']): A social media algorithm promotes content that reinforces traditional gender roles. What does this p... — Issues: anachronism:algorithm_in_EP1, anachronism:social media_in_EP1
- **ID 112** (['EP1']): A tech company's product team designs a pregnancy app without consulting the people who will use it.... — Issues: anachronism:tech company_in_EP1
- **ID 159** (['EP1']): Why do food deserts persist in wealthy nations?... — Issues: anachronism:food desert_in_EP1
- **ID 169** (['EP1']): Why has private equity purchased so many hospitals and nursing homes?... — Issues: anachronism:private equity_in_EP1, anachronism:nursing home_in_EP1
- **ID 196** (['EP1']): A tech company's workforce skews younger as older workers are displaced by AI. What does this demogr... — Issues: anachronism:tech company_in_EP1
- **ID 300** (['EP2']): What does the rise of creator economies and influencer culture tell us about the commodification of ... — Issues: anachronism:influencer_in_EP2
- **ID 483** (['EP2']): A neighborhood's food desert overlaps with a high pollution zone. How might we understand the relati... — Issues: anachronism:food desert_in_EP2

**Total anachronistic questions: 22**

## Grammar Issues

Found 37 questions with grammar issues:

- **ID 98**: Why do certain ethnic groups are disproportionately funneled into prison while similar behaviors by other groups are tre...
- **ID 103**: A company's parental leave policy gives more weeks to birthing parents than to adopting parents. What does this distinct...
- **ID 123**: Why do care workers who enable other people to participate in the formal economy are themselves paid below a living wage...
- **ID 124**: Why do immigrant women are disproportionately concentrated in the lowest-paid care work while immigrant men are concentr...
- **ID 178**: A city's public buildings are fully accessible but the surrounding sidewalks are not. How might we understand the relati...
- **ID 181**: Why do disabled people are overrepresented in the prison system and underrepresented in disability support services?...
- **ID 195**: A healthcare system prioritizes treatments for younger patients over palliative care for the elderly. How might we under...
- **ID 228**: How did religious communities in pre-industrial Europe provide care for the poor and sick before formal welfare systems?...
- **ID 321**: Why do labor market policies that focus on matching workers to jobs ignore the power dynamics that determine which jobs ...
- **ID 350**: Why do women in the Global North benefit from the cheap care labor of women in the Global South, creating a global hiera...
- **ID 351**: Why is care work — childcare, eldercare, domestic labor — typically unpaid or underpaid compared to other labor?...
- **ID 413**: Why do countries that were colonized for their resources tend to have economies that are structured to benefit foreign i...
- **ID 480**: A family's time poverty increases as both parents work multiple jobs. How might we understand the relationship between i...
- **ID 492**: A workplace's wellness program offers yoga classes during lunch but not childcare during those same hours. How might we ...
- **ID 504**: A pharmaceutical company argues that high drug prices are necessary to fund research and development. But the company sp...
- **ID 523**: How would an analysis that examines who benefits from the current healthcare system differ from one that examines why he...
- **ID 531**: The move toward asset-based welfare — where retirement security depends on market performance — has transferred risk fro...
- **ID 609**: Why do family policies that assume a two-parent household fail to address the realities of single-parent families, which...
- **ID 612**: Why do white-collar professions that are dominated by women consistently pay less than male-dominated professions requir...
- **ID 677**: A family's elder care responsibilities fall disproportionately on women. How might we understand the relationship betwee...

## DM Terminology Violations

Found 11 questions containing DM terminology (should be ideologically neutral):

- **ID 117** (Type A): How did the division between productive and reproductive labor in pre-industrial societies shape gen... — Terms: dm_terms:reproductive labor
- **ID 273** (Type D): How does the reserve army of labor function to discipline workers and suppress wages?... — Terms: dm_terms:reserve army of labor
- **ID 317** (Type B): When we examine the explanation for gender inequality that emphasizes cultural norms versus the one ... — Terms: dm_terms:reproductive labor
- **ID 517** (Type D): How does the concept of surplus value explain the relationship between workers and employers?... — Terms: dm_terms:surplus value
- **ID 518** (Type D): How does the concept of the reserve army of labor explain the persistence of unemployment?... — Terms: dm_terms:reserve army of labor
- **ID 520** (Type D): How does the reserve army of labor function as a mechanism of wage discipline?... — Terms: dm_terms:reserve army of labor
- **ID 546** (Type D): What is commodity fetishism and how does it shape our understanding of economic relationships?... — Terms: dm_terms:commodity fetishism
- **ID 848** (Type D): How does the division between productive and reproductive labor shape the distribution of wealth in ... — Terms: dm_terms:reproductive labor
- **ID 875** (Type D): How does the concept of social reproduction explain the relationship between paid work and unpaid ca... — Terms: dm_terms:social reproduction
- **ID 919** (Type D): What is primitive accumulation and how does it explain the historical origins of capitalist property... — Terms: dm_terms:primitive accumulation
- **ID 1125** (Type D): How does the concept of social reproduction theory explain the connection between paid work and unpa... — Terms: dm_terms:social reproduction

## Tag Combination Analysis

Most common axis1+axis2 combinations (potential redundancy):

| Axis 1 | Axis 2 | Count |
|---|---|---|
| ('A1',) | ('EP1',) | 34 |
| ('A1',) | ('EP4',) | 30 |
| ('A1',) | ('EP3',) | 29 |
| ('A1',) | ('EP5',) | 27 |
| ('A1',) | ('EP6',) | 27 |
| ('A1',) | ('EP2',) | 24 |
| ('A1', 'D1') | ('EP6',) | 19 |
| ('A1', 'D1') | ('EP3',) | 18 |
| ('F3',) | ('EP5',) | 16 |
| ('A1', 'D1') | ('EP5',) | 15 |
| ('D2',) | ('EP1',) | 12 |
| ('C3',) | ('EP3',) | 12 |
| ('D3',) | ('EP1',) | 11 |
| ('A1', 'D1') | ('EP2',) | 11 |
| ('D4',) | ('EP2',) | 11 |
| ('A1', 'F4') | ('EP5',) | 11 |
| ('A1', 'D1') | ('EP1',) | 10 |
| ('F3',) | ('EP2',) | 9 |
| ('A1', 'D1') | ('EP4',) | 9 |
| ('C3',) | ('EP4',) | 9 |

## Recommendations

The dataset is of generally good quality. Focus revision efforts on the bottom 10% of questions.

### Priority Actions

1. **Quality fixes needed**: 4 questions score below 4.0 (grammar, vagueness, DM terminology)
2. **Coherence fixes needed**: 1 questions score below 4.0 (anachronisms, tag mismatches)
3. **Uniqueness concerns**: 0 questions score below 3.0 (highly similar to others)
4. **Epoch anachronisms**: 22 questions have content inconsistent with their epoch tag
5. **Grammar issues**: 37 questions have subject-verb agreement or other grammar problems
