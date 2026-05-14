"""
Question Generation Script

Generates unique, hand-crafted questions for DM-aligned training data.
Questions are inspired by the topic taxonomy (Axis 1 categories × Axis 2 epochs)
but are not templated — each question is a distinct, concrete question
that anyone might naturally ask.

Type distribution target: A=40%, B=20%, C=20%, D=5%, E=15%
Cross-domain target: ≥20%
Multi-tag (≥2 axis1) target: ≥30%

Usage:
    python -m src.teacher.generate_questions --batch 1 --output data/raw/questions.json
    python -m src.teacher.generate_questions --batch 2 --output data/raw/questions.json
    ...
    python -m src.teacher.generate_questions --batch 6 --output data/raw/questions.json

Each batch generates 250 questions (except batch 6 which generates 15 to hit 1500 total).
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Type distribution per batch of 250 ──
# A: 100, B: 50, C: 50, D: 12, E: 38
BATCH_DISTRIBUTION = {
    "A": 100,
    "B": 50,
    "C": 50,
    "D": 12,
    "E": 38,
}

# ── Question pools ──
# Each pool is a list of (question_text, axis1_tags, axis2_tags, cross_domain)
# These are hand-crafted, unique questions inspired by the topic taxonomy.

# ──────────────────────────────────────────────────────────────────────
# TYPE A: Neutral Framing Questions (~40%)
# Everyday questions where the default AI answer is liberal-reformist,
# but a DM answer provides deeper structural analysis.
# ──────────────────────────────────────────────────────────────────────

TYPE_A_POOLS = [
    # ── Class & Labor (A) ──
    ("Why do companies with record profits still refuse to hire more workers?", ["A1"], ["EP6"], False),
    ("What explains the gap between corporate revenue growth and worker pay over the past three decades?", ["A1"], ["EP5", "EP6"], False),
    ("Why do some workers have to choose between buying groceries and paying rent while nearby stores sell luxury goods?", ["A1", "A5"], ["EP6"], False),
    ("How does the expectation that employees be 'passionate' about their work affect wage negotiations?", ["A1", "E4"], ["EP6"], False),
    ("Why do temp agencies and staffing firms keep growing even when job openings are high?", ["A2", "A5"], ["EP6"], False),
    ("What happens to the communities left behind when factories move overseas?", ["A1", "J1", "F3"], ["EP5"], False),
    ("Why do gig economy workers get classified as independent contractors when the platforms control their rates and schedules?", ["A1", "A5"], ["EP6"], False),
    ("How does the two-job economy affect family life and community participation?", ["A1", "D1", "D6"], ["EP6"], False),
    ("Why do workers in essential industries like food service and home care earn so little compared to their social value?", ["A1", "A5"], ["EP6"], False),
    ("What explains the rise of 'gig' work in fields like tutoring, design, and writing that used to be salaried?", ["A1", "A2"], ["EP6"], False),
    ("Why do warehouse workers face intense monitoring while executives work remotely?", ["A1", "E1"], ["EP6"], False),
    ("How does the shift from pensions to 401(k)s change the relationship between workers and their employers?", ["A1", "G4"], ["EP5"], False),
    ("Why do some companies invest millions in workplace wellness programs while keeping wages below a living standard?", ["A1", "D3"], ["EP6"], False),
    ("What happens to workers' bargaining power when their skills become automated?", ["A1", "A2"], ["EP6"], False),
    ("Why do fast-food workers rely on food stamps while their corporations post record profits?", ["A1", "A5", "D5"], ["EP6"], False),
    ("How does the gig economy's rating system function as a form of managerial control?", ["A1", "A2"], ["EP6"], False),
    ("Why do companies prefer hiring contractors over full-time employees even for core business functions?", ["A1", "A2"], ["EP6"], False),
    ("What explains the decline of workplace safety protections as employment becomes more precarious?", ["A1", "D3"], ["EP6"], False),
    ("How does wage theft disproportionately affect immigrant and minority workers?", ["A1", "B6", "H1"], ["EP6"], False),
    ("Why do unpaid internships persist as a gateway to professional careers?", ["A1", "D4", "B7"], ["EP6"], False),

    # ── Race & Racialization (B) ──
    ("Why do neighborhoods with similar crime rates have very different police response times?", ["B1", "J4"], ["EP6"], False),
    ("What explains the wealth gap between families who bought homes in the 1970s and those trying today?", ["B3", "G3"], ["EP4", "EP5"], False),
    ("Why are certain names on resumes less likely to get callbacks in the same job market?", ["B7"], ["EP6"], False),
    ("How does the school-to-prison pipeline differ between predominantly white and predominantly minority districts?", ["B3", "D4"], ["EP6"], False),
    ("Why do food banks cluster in certain neighborhoods while grocery stores avoid them?", ["B1", "J3", "D5"], ["EP6"], False),
    ("What happens to communities when a large employer moves to a state with right-to-work laws?", ["B3", "A2"], ["EP5"], False),
    ("Why do environmental hazards like landfills and highways tend to be located near minority neighborhoods?", ["B3", "J3"], ["EP6"], False),
    ("How does the criminalization of drug use differ between communities that use and communities that don't?", ["B3", "B1"], ["EP5"], False),
    ("Why do some cities invest in beautification projects that displace the very residents who built the community's culture?", ["B3", "J5"], ["EP6"], False),
    ("What explains the concentration of predatory lenders in certain zip codes?", ["B3", "B7"], ["EP6"], False),
    ("How does the fashion industry's use of racialized aesthetics intersect with the compensation of the models and creators?", ["B3", "C5"], ["EP6"], False),
    ("Why do certain sports franchises move between cities, and who benefits when they do?", ["B3", "J5"], ["EP6"], False),
    ("How does the labeling of certain foods as 'ethnic' affect their pricing and who gets to profit?", ["B3", "D5"], ["EP6"], False),
    ("What happens to Indigenous communities when their water sources are contaminated by nearby industrial operations?", ["B4", "F2", "J3"], ["EP6"], False),
    ("Why do some countries' cultural products dominate global markets while others are marginalized?", ["B3", "F4"], ["EP6"], False),

    # ── Gender & Sexuality (C) ──
    ("Why do women in their 40s face a wage gap increase precisely when they're also caring for aging parents?", ["C2", "G2"], ["EP6"], False),
    ("How does the design of workplace schedules reflect assumptions about who handles care work at home?", ["C5", "D1"], ["EP6"], False),
    ("Why do some countries allow pregnancy discrimination while others prohibit it, and what does that say about the purpose of employment law?", ["C6"], ["EP4", "EP5"], False),
    ("What happens to domestic workers when immigration policies change?", ["C5", "H1"], ["EP6"], False),
    ("How does the beauty industry profit from insecurities that it helps create across different age groups?", ["C5", "D6"], ["EP6"], False),
    ("Why do reproductive health services cluster in certain regions while being unavailable in others?", ["C6", "J1"], ["EP6"], False),
    ("How does the concept of 'maternal instinct' affect the valuation of care work in society?", ["C1", "D1"], ["EP6"], False),
    ("Why do transgender people face higher rates of housing insecurity and employment discrimination?", ["C3"], ["EP6"], False),
    ("What explains the gap between the portrayal of women in advertising and the actual wage gap?", ["C5", "B7"], ["EP6"], False),
    ("How does the sexual division of labor show up in the automation of different types of work?", ["C5", "A1"], ["EP6"], False),
    ("Why do some cultures pathologize certain family structures while normalizing others?", ["C1", "I1"], ["EP6"], False),
    ("How does the commercialization of romance and dating apps affect relationship dynamics and economic dependency?", ["C5", "D1"], ["EP6"], False),
    ("What happens to women's economic security when reproductive technologies become privatized?", ["C6", "D3"], ["EP6"], False),
    ("Why do care professions like nursing and teaching remain female-dominated and underpaid?", ["C5", "D1"], ["EP6"], False),
    ("How does the policing of transgender bathrooms reinforce broader patterns of gender control?", ["C3", "B7"], ["EP6"], False),

    # ── Social Reproduction (D) ──
    ("Why do parents in dual-income households still report feeling like they can't 'have it all'?", ["D1", "D6"], ["EP6"], False),
    ("How does the quality of a child's early education depend on their parents' employment situation?", ["D4", "D1"], ["EP6"], False),
    ("Why do some cities have park access while others don't, and how does that affect health outcomes?", ["D2", "J1"], ["EP6"], False),
    ("What happens to communities when public libraries face budget cuts?", ["D4", "D1"], ["EP6"], False),
    ("How does the cost of childcare shape women's career trajectories and lifetime earnings?", ["D1", "C2"], ["EP6"], False),
    ("Why do some neighborhoods have adequate grocery stores while others rely on convenience stores?", ["D5", "J1"], ["EP6"], False),
    ("How does the design of public transportation affect people's ability to access jobs, healthcare, and education?", ["D2", "D3", "D4"], ["EP6"], False),
    ("Why do food prices rise faster than wages, and who profits from the difference?", ["D5", "A1"], ["EP6"], False),
    ("What explains the difference in life expectancy between neighborhoods just miles apart?", ["D3", "J3"], ["EP6"], False),
    ("How does the privatization of water services affect low-income households?", ["D2", "D3"], ["EP6"], False),
    ("Why do elderly people in some countries age with dignity while others face institutional neglect?", ["G2", "D3"], ["EP6"], False),
    ("How does the cost of raising a child compare to the economic benefits that child provides to society?", ["D1", "D4"], ["EP6"], False),
    ("Why do some employers offer 'family-friendly' policies that are difficult to actually use without career penalty?", ["D1", "C5"], ["EP6"], False),
    ("How does the housing market's preference for single-family zoning affect families with children?", ["D2", "D4"], ["EP6"], False),
    ("What happens to communities when their only hospital closes due to profitability concerns?", ["D3", "J1"], ["EP6"], False),

    # ── Disability & Ableism (E) ──
    ("Why do workplaces that advertise diversity often exclude people with visible disabilities?", ["E2", "E4"], ["EP6"], False),
    ("How does the productivity norm shape who is considered 'valuable' in the labor market?", ["E1", "A1"], ["EP6"], False),
    ("Why do assistive technologies that could improve millions of lives remain expensive and inaccessible?", ["E2", "D3"], ["EP6"], False),
    ("How does the design of public spaces reflect assumptions about which bodies are 'normal'?", ["E2", "J1"], ["EP6"], False),
    ("What happens to people with intellectual disabilities when family caregivers age or pass away?", ["E3", "G2"], ["EP6"], False),
    ("Why do mental health conditions remain stigmatized in workplace culture despite growing awareness?", ["E4", "D3"], ["EP6"], False),
    ("How does the education system's emphasis on standardized testing disadvantage neurodivergent students?", ["E5", "D4"], ["EP6"], False),
    ("Why do insurance companies cover some disability treatments but not others?", ["E3", "D3"], ["EP6"], False),
    ("How does the concept of 'reasonable accommodation' function in practice for disabled workers?", ["E2", "A1"], ["EP6"], False),
    ("What explains the overrepresentation of disabled people in poverty and underrepresentation in leadership?", ["E1", "E3"], ["EP6"], False),

    # ── Coloniality & Indigeneity (F) ──
    ("Why do former colonies often struggle with debt to the countries that colonized them?", ["F3", "G4"], ["EP2", "EP5"], False),
    ("How does the extraction of rare earth minerals for technology connect consumers in wealthy countries to labor conditions abroad?", ["F3", "A1"], ["EP6"], False),
    ("Why do some countries protect indigenous land rights while others prioritize resource extraction?", ["F2", "F3"], ["EP6"], False),
    ("How does the global water trade affect communities in water-scarce regions?", ["F3", "D2"], ["EP6"], False),
    ("What happens to local economies when multinational corporations extract resources without reinvesting?", ["F3", "J1"], ["EP6"], False),
    ("Why do cultural artifacts from colonized countries remain in museums of former colonial powers?", ["F4", "I1"], ["EP2", "EP6"], False),
    ("How does the concept of 'free trade' affect countries with weaker bargaining positions?", ["F3", "J2"], ["EP5"], False),
    ("What explains the persistence of debt traps for developing nations that borrow from international financial institutions?", ["F3", "G4"], ["EP5"], False),
    ("How does the global migration of care workers from the Global South to the North affect sending communities?", ["F3", "H4", "D1"], ["EP6"], False),
    ("Why do some countries resist international climate agreements that require emission cuts?", ["F3", "J2"], ["EP6"], False),

    # ── Age & Generational (G) ──
    ("Why do young people today face higher education costs and housing prices than their parents did at the same age?", ["G1", "D4", "D2"], ["EP6"], False),
    ("How does the shift from defined-benefit to defined-contribution pensions affect retirement security?", ["G4", "G2"], ["EP5"], False),
    ("Why do older workers face age discrimination even when they have valuable experience?", ["G2", "A1"], ["EP6"], False),
    ("What happens to intergenerational wealth transfer when housing prices outpace income growth?", ["G3", "G1"], ["EP6"], False),
    ("How does the student debt burden affect young people's ability to start families or buy homes?", ["G1", "D4", "D2"], ["EP6"], False),
    ("Why do some countries have generous elder care systems while others leave it to families?", ["G2", "D3"], ["EP6"], False),
    ("How does the timing of retirement policies affect different generations differently?", ["G4", "G3"], ["EP6"], False),
    ("What explains the growing political influence of older voters compared to younger ones?", ["G1", "G2"], ["EP6"], False),

    # ── Immigration (H) ──
    ("Why do countries that benefit from immigrant labor often restrict immigration policies?", ["H1", "H3"], ["EP6"], False),
    ("How does the border industrial complex affect communities on both sides of international boundaries?", ["H2", "F5"], ["EP6"], False),
    ("What happens to families when one parent is deported while the other remains?", ["H3", "D1"], ["EP6"], False),
    ("How does the brain drain from developing countries affect their healthcare and education systems?", ["H4", "D3"], ["EP6"], False),
    ("Why do refugee resettlement programs vary so dramatically between countries?", ["H5", "F5"], ["EP6"], False),
    ("How does the documentation status of workers affect their ability to report workplace violations?", ["H3", "A1"], ["EP6"], False),
    ("What explains the difference in treatment between skilled and unskilled migrants in immigration policy?", ["H1", "H4"], ["EP6"], False),
    ("How do remittances shape the economies of sending countries while exploiting the workers who send them?", ["H1", "F3"], ["EP6"], False),

    # ── Religion (I) ──
    ("Why do some religious organizations accumulate vast wealth while advocating for the poor?", ["I1", "A1"], ["EP6"], False),
    ("How does the secular framing of public policy affect communities whose values are rooted in religious traditions?", ["I2", "I4"], ["EP6"], False),
    ("What happens to faith-based social services when government funding changes?", ["I3", "D3"], ["EP6"], False),
    ("How does the rise of prosperity gospel reflect broader economic ideologies?", ["I1", "B7"], ["EP6"], False),
    ("Why do some countries enforce secularism more strictly than others, and who benefits from that enforcement?", ["I2", "I4"], ["EP6"], False),

    # ── Geography & Spatial Power (J) ──
    ("Why do cities that attract tech companies see housing prices soar while surrounding regions stagnate?", ["J1", "J5"], ["EP6"], False),
    ("How does the location of polluting industries relate to the political power of affected communities?", ["J3", "J4"], ["EP6"], False),
    ("What explains the difference in internet access between urban and rural areas?", ["J1", "D4"], ["EP6"], False),
    ("How does the concentration of wealth in global cities affect the rest of the country?", ["J1", "J2"], ["EP6"], False),
    ("Why do some regions become 'rust belts' while others thrive, and is this inevitable?", ["J1", "J5"], ["EP3", "EP5"], False),
    ("How does the design of suburban sprawl affect social cohesion and environmental sustainability?", ["J1", "J5"], ["EP4"], False),
    ("What happens to communities when a major employer leaves and the tax base shrinks?", ["J1", "A2"], ["EP5"], False),
    ("Why do some neighborhoods become 'safe' investments for developers while others are labeled 'high risk'?", ["J4", "B3"], ["EP6"], False),

    # ── Intersectional Identities (K) ──
    ("How do the economic challenges faced by disabled women of color differ from those faced by other groups?", ["K8", "E2", "C2"], ["EP6"], False),
    ("Why do Indigenous women face higher rates of violence and economic marginalization?", ["K2", "B4", "C2"], ["EP6"], False),
    ("How does the experience of queer migrants differ from that of straight migrants in terms of labor market access?", ["K5", "C3", "H1"], ["EP6"], False),
    ("What explains the economic vulnerability of young Black men in urban areas?", ["K6", "B1", "A2"], ["EP6"], False),
    ("How do working-class mothers of color navigate the intersection of wage work and unpaid care responsibilities?", ["K7", "C2", "B3", "D1"], ["EP6"], False),
    ("Why do elderly poor people of color face worse health outcomes than their white counterparts?", ["K4", "E1", "B3"], ["EP6"], False),
    ("How do disabled migrants access (or fail to access) disability services across different countries?", ["K3", "E3", "H3"], ["EP6"], False),
    ("What happens to Black trans women who age out of survival economies into older age?", ["K1", "E3", "G2"], ["EP6"], False),
    ("How do the economic pressures on immigrant mothers differ from those on non-immigrant mothers?", ["K7", "H1", "C2"], ["EP6"], False),
    ("Why do disabled women of color face higher rates of forced sterilization in some countries?", ["K8", "E2", "C6"], ["EP6"], False),

    # ── Pre-Capitalist (EP1) ──
    ("How did the transition from feudalism to capitalism change the relationship between workers and the products of their labor?", ["A1", "F1"], ["EP1", "EP2"], False),
    ("What happened to communal land systems as private property became the dominant model?", ["F1", "J5"], ["EP1", "EP2"], False),
    ("How did the enclosure movements in England displace rural populations and create a wage-dependent class?", ["A2", "F1"], ["EP2"], False),
    ("Why did slave-based economies persist for centuries despite being economically inefficient by some measures?", ["A1", "B1"], ["EP2"], False),
    ("How did guild systems regulate labor before the rise of factory capitalism?", ["A1", "A4"], ["EP1"], False),
    ("What role did indigenous communal systems play in shaping early colonial economies?", ["F2", "F4"], ["EP2"], False),
    ("How did the transatlantic slave trade restructure labor systems across three continents?", ["A1", "B1", "F3"], ["EP2"], False),
    ("Why did serfdom persist in some regions of Europe longer than in others?", ["A1", "J1"], ["EP1", "EP2"], False),
    ("How did the colonization of the Americas transform global patterns of production and exchange?", ["F3", "F1"], ["EP2"], False),
    ("What happened to women's economic roles during the transition from household production to factory labor?", ["C5", "A1"], ["EP2", "EP3"], False),
]

# ──────────────────────────────────────────────────────────────────────
# TYPE B: Contrast Questions (~20%)
# Questions that ask for analysis from different angles, revealing
# what different approaches make visible and invisible.
# ──────────────────────────────────────────────────────────────────────

TYPE_B_POOLS = [
    ("What does the focus on 'personal responsibility' for health outcomes overlook about the conditions that shape health?", ["D3", "B7"], ["EP6"], False),
    ("Why do corporate diversity statements not correlate with changes in workforce demographics?", ["B7", "C5"], ["EP6"], False),
    ("What does the narrative of 'rising tide lifts all boats' miss about how wealth actually circulates?", ["A1", "B7"], ["EP6"], False),
    ("Why do community policing initiatives often fail to rebuild trust in the communities they're meant to serve?", ["B3", "J4"], ["EP6"], False),
    ("What does treating homelessness as a housing shortage overlook about the role of mental health and addiction services?", ["D2", "D3", "E4"], ["EP6"], False),
    ("Why do carbon pricing mechanisms not address the structural growth imperative that drives overconsumption?", ["D2", "B7"], ["EP6"], False),
    ("What does the emphasis on 'work-life balance' miss about the structural demands of precarity?", ["D1", "D6", "A1"], ["EP6"], False),
    ("Why do scholarship programs for underrepresented students not address the structural barriers within institutions?", ["D4", "B3"], ["EP6"], False),
    ("What does the framing of 'affordable housing' as a supply problem overlook about land speculation?", ["D2", "J5"], ["EP6"], False),
    ("Why do financial literacy programs not reduce debt levels in low-income communities?", ["D3", "B7"], ["EP6"], False),
    ("What does the focus on 'quality of life' crime metrics miss about the conditions that produce crime?", ["B3", "J4"], ["EP6"], False),
    ("Why do employee stock ownership plans not change the fundamental power dynamics in companies?", ["A1", "B7"], ["EP6"], False),
    ("What does the idea of 'social mobility' overlook about the structural barriers to upward movement?", ["B7", "G3"], ["EP6"], False),
    ("Why do environmental justice campaigns that focus on individual polluters not address systemic patterns?", ["J3", "B3"], ["EP6"], False),
    ("What does treating poverty as a 'gap' to be filled overlook about the mechanisms that produce it?", ["A1", "B7"], ["EP6"], False),
    ("Why do corporate sustainability reports not correlate with actual environmental performance?", ["B7", "D2"], ["EP6"], False),
    ("What does the emphasis on 'first-generation college students' miss about the structural advantages that all students bring?", ["D4", "B7"], ["EP6"], False),
    ("Why do anti-trafficking campaigns not address the economic conditions that make migration necessary?", ["H1", "C5"], ["EP6"], False),
    ("What does the narrative of 'giving back to the community' overlook about who benefits from corporate philanthropy?", ["A1", "B7"], ["EP6"], False),
    ("Why do restorative justice programs succeed in some contexts and fail in others?", ["B3", "J4"], ["EP6"], False),
    ("What does the focus on 'accessible housing' overlook about the economic forces that make housing unaffordable?", ["D2", "J5"], ["EP6"], False),
    ("Why do workplace mental health programs not reduce the prevalence of burnout?", ["E4", "D3"], ["EP6"], False),
    ("What does the framing of 'elder care crisis' as a demographic problem miss about the organization of care work?", ["G2", "D1", "C5"], ["EP6"], False),
    ("Why do food donation programs not address the structural causes of food insecurity?", ["D5", "B7"], ["EP6"], False),
    ("What does the idea of 'inclusive growth' overlook about the distributional dynamics of economic expansion?", ["A1", "B7"], ["EP6"], False),
    ("Why do paid family leave policies not achieve gender equality in care responsibilities?", ["D1", "C5"], ["EP6"], False),
    ("What does the focus on 'digital divide' overlook about the economic incentives that shape technology access?", ["D4", "J1"], ["EP6"], False),
    ("Why do community land trusts succeed in some cities and not others?", ["D2", "J5"], ["EP6"], False),
    ("What does treating crime prevention as a technology problem miss about the social conditions that produce crime?", ["B3", "J4"], ["EP6"], False),
    ("Why do corporate anti-harassment training programs not eliminate harassment?", ["C5", "A1"], ["EP6"], False),
    ("What does the emphasis on 'economic opportunity' overlook about the structural barriers to seizing opportunity?", ["B7", "A1"], ["EP6"], False),
    ("Why do public health campaigns about obesity not address food environment inequities?", ["D5", "J3"], ["EP6"], False),
    ("What does the framing of 'aging population' as a crisis miss about the value of intergenerational knowledge?", ["G2", "D1"], ["EP6"], False),
    ("Why do microenterprise programs not lift entrepreneurs out of poverty at scale?", ["A1", "B7"], ["EP6"], False),
    ("What does the focus on 'green jobs' overlook about the conditions under which those jobs are performed?", ["A1", "D2"], ["EP6"], False),
    ("Why do teacher bonus programs not improve educational outcomes in underfunded districts?", ["D4", "B3"], ["EP6"], False),
    ("What does the narrative of 'aging in place' overlook about the infrastructure needs of older adults?", ["G2", "D2"], ["EP6"], False),
    ("Why do community benefit agreements often fail to deliver promised benefits?", ["J5", "B3"], ["EP6"], False),
    ("What does treating addiction as a public health crisis miss about the role of economic despair?", ["D3", "E4"], ["EP6"], False),
    ("Why do equity-focused hiring initiatives often result in 'diversity wash' without structural change?", ["B3", "C5"], ["EP6"], False),
    ("What does the emphasis on 'student success' overlook about the structural conditions that produce failure?", ["D4", "B7"], ["EP6"], False),
    ("Why do community gardens not address the structural causes of food deserts?", ["D5", "J1"], ["EP6"], False),
    ("What does the framing of 'workforce development' miss about the quality of jobs being developed?", ["A1", "D4"], ["EP6"], False),
    ("Why do wellness retreats and self-care industries thrive while public health underfunded?", ["D3", "D6"], ["EP6"], False),
    ("What does the idea of 'community resilience' overlook about the structural conditions that undermine resilience?", ["J1", "B7"], ["EP6"], False),
    ("Why do mentorship programs for underrepresented groups not change institutional cultures?", ["D4", "B3"], ["EP6"], False),
    ("What does the focus on 'accessibility' overlook about the economic barriers to accessibility?", ["E2", "D3"], ["EP6"], False),
    ("Why do public-private partnerships for affordable housing produce far fewer units than promised?", ["D2", "J5"], ["EP6"], False),
    ("What does the narrative of 'giving back' in corporate social responsibility miss about power and accountability?", ["A1", "B7"], ["EP6"], False),

    # ── Cross-domain Type B ──
    ("What does the focus on 'user experience' in software design overlook about the labor conditions of content moderators?", ["A1", "E4"], ["EP6"], True),
    ("Why do algorithmic recommendations not improve discovery but instead reinforce existing patterns?", ["B7", "A1"], ["EP6"], True),
    ("What does the framing of 'content moderation' as a design problem miss about the labor and political dimensions?", ["A1", "C3"], ["EP6"], True),
    ("Why do streaming platforms' recommendation algorithms not promote diverse content despite claims of supporting creators?", ["A1", "C5"], ["EP6"], True),
    ("What does the narrative of 'democratizing creativity' through AI tools overlook about the training data and labor behind those tools?", ["A1", "F4"], ["EP6"], True),
    ("Why do social media platforms' community guidelines not address structural power imbalances in content visibility?", ["A1", "B7"], ["EP6"], True),
    ("What does the focus on 'platform governance' overlook about the ownership and profit structure of social media?", ["A1", "B7"], ["EP6"], True),
    ("Why do e-sports tournaments offer prize money that dwarfs salaries in many traditional sports, and who benefits?", ["A1", "A3"], ["EP6"], True),
    ("What does the framing of 'creator economy' miss about the dependency relationship between creators and platforms?", ["A1", "A5"], ["EP6"], True),
    ("Why do video game microtransactions target psychologically vulnerable players disproportionately?", ["E4", "A1"], ["EP6"], True),
    ("What does the narrative of 'open innovation' overlook about the proprietary control of research outcomes?", ["A1", "F4"], ["EP6"], True),
    ("Why do music streaming platforms pay artists so little while platform valuations grow?", ["A1", "C5"], ["EP6"], True),
    ("What does the focus on 'accessibility' in app design overlook about the economic barriers to owning devices?", ["E2", "A1"], ["EP6"], True),
    ("Why do fitness tracking apps promote individual responsibility while ignoring the structural determinants of health?", ["D3", "B7"], ["EP6"], True),
    ("What does the framing of 'sports business' as entertainment miss about the labor exploitation in athlete development?", ["A1", "A3"], ["EP6"], True),

    # ── Epoch-specific Type B ──
    ("What does the post-war welfare state settlement miss about who was excluded from its benefits?", ["B3", "C5"], ["EP4"], False),
    ("Why did the neoliberal turn reframe social problems as individual failures?", ["B7", "A1"], ["EP5"], False),
    ("What does the industrial revolution's narrative of 'progress' overlook about the conditions of factory workers?", ["A1", "E1"], ["EP3"], False),
    ("Why did the colonial extraction of resources create lasting developmental gaps?", ["F3", "J2"], ["EP2"], False),
    ("What does the Cold War framing of economic systems overlook about class dynamics within countries?", ["A1", "B7"], ["EP4"], False),
]

# ──────────────────────────────────────────────────────────────────────
# TYPE C: Application Questions (~20%)
# Current events or concrete phenomena that invite structural analysis.
# ──────────────────────────────────────────────────────────────────────

TYPE_C_POOLS = [
    ("Why has the price of prescription drugs increased faster than inflation for decades?", ["A1", "D3"], ["EP6"], False),
    ("What explains the consolidation of the airline industry and its effect on ticket prices?", ["A1", "B7"], ["EP5", "EP6"], False),
    ("Why have warehouse automation investments grown while warehouse worker injuries remain high?", ["A1", "E1"], ["EP6"], False),
    ("What's behind the surge in private equity ownership of nursing homes and its effect on care quality?", ["D3", "G2"], ["EP6"], False),
    ("Why have rental prices outpaced income growth in almost every major city?", ["D2", "J5"], ["EP6"], False),
    ("What explains the rapid growth of the for-profit higher education sector?", ["D4", "A1"], ["EP5", "EP6"], False),
    ("Why have commodity futures speculation increased the volatility of food prices?", ["D5", "A1"], ["EP6"], False),
    ("What's behind the consolidation of the meatpacking industry and its effect on workers and consumers?", ["A1", "D5"], ["EP5", "EP6"], False),
    ("Why have hospital mergers increased prices without improving outcomes?", ["D3", "A1"], ["EP6"], False),
    ("What explains the boom in single-family home rentals by institutional investors?", ["D2", "A1"], ["EP6"], False),
    ("Why have agricultural subsidies disproportionately benefited large agribusiness over small farmers?", ["D5", "A1"], ["EP5"], False),
    ("What's behind the growth of the surveillance technology industry and its expansion into schools and public spaces?", ["A1", "J4"], ["EP6"], False),
    ("Why have pharmaceutical companies invested more in 'me-too' drugs than in novel treatments?", ["D3", "A1"], ["EP6"], False),
    ("What explains the concentration of media ownership and its effect on news coverage?", ["A1", "B7"], ["EP6"], False),
    ("Why have logistics companies expanded while their workers face increasing precarity?", ["A1", "A5"], ["EP6"], False),
    ("What's behind the growth of the for-profit prison industry and its lobbying efforts?", ["A1", "D3"], ["EP5", "EP6"], False),
    ("Why have insurance premiums for health, home, and auto all increased simultaneously?", ["A1", "D3"], ["EP6"], False),
    ("What explains the rise of buy-now-pay-later services and their impact on household debt?", ["A1", "G4"], ["EP6"], False),
    ("Why have broadband providers faced little competition despite being natural monopolies?", ["A1", "J1"], ["EP6"], False),
    ("What's behind the consolidation of the seed and pesticide industry and its effect on farmers?", ["D5", "A1"], ["EP5", "EP6"], False),
    ("Why have university endowments grown while state funding for public education declined?", ["D4", "A1"], ["EP5", "EP6"], False),
    ("What explains the growth of the gig economy in caregiving roles like elder care and childcare?", ["A5", "D1"], ["EP6"], False),
    ("Why have commercial property values declined while industrial warehouse values surged?", ["J5", "A1"], ["EP6"], False),
    ("What's behind the rise of corporate lobbying spending and its effect on regulatory policy?", ["A1", "B7"], ["EP6"], False),
    ("Why have patent thickets in biotechnology slowed rather than accelerated drug development?", ["D3", "A1"], ["EP6"], False),
    ("What explains the growth of the debt collection industry and its impact on low-income households?", ["A1", "G4"], ["EP6"], False),
    ("Why have grocery chains consolidated to the point where a handful of companies control most market share?", ["A1", "D5"], ["EP6"], False),
    ("What's behind the increase in corporate stock buybacks and their effect on investment and wages?", ["A1", "B7"], ["EP6"], False),
    ("Why have water privatization projects often led to price increases for consumers?", ["D2", "A1"], ["EP5", "EP6"], False),
    ("What explains the growth of the private student loan industry and its lending practices?", ["D4", "A1"], ["EP5", "EP6"], False),
    ("Why have pharmaceutical companies acquired and shelved competitors' drugs?", ["D3", "A1"], ["EP6"], False),
    ("What's behind the expansion of the surveillance capitalist business model into new domains?", ["A1", "E4"], ["EP6"], True),
    ("Why have music royalties shifted from album sales to streaming, and who benefits?", ["A1", "C5"], ["EP6"], True),
    ("What explains the rise of influencer marketing and its effect on advertising and consumer behavior?", ["A1", "C5"], ["EP6"], True),
    ("Why have video game companies shifted from one-time purchases to live-service models?", ["A1", "C5"], ["EP6"], True),
    ("What's behind the consolidation of sports media rights and its effect on fans?", ["A3", "A1"], ["EP6"], True),
    ("Why have fitness companies shifted from gym memberships to subscription-based digital content?", ["A1", "D3"], ["EP6"], True),
    ("What explains the growth of the pet industry and its premium pricing despite lower social necessity?", ["A1", "D5"], ["EP6"], True),
    ("Why do technology companies invest heavily in AI while underinvesting in maintenance and repair labor?", ["A1", "E1"], ["EP6"], True),
    ("What's behind the trend of sports franchises moving to new cities with public subsidies?", ["A3", "J5"], ["EP6"], True),
    ("Why have film studios increasingly relied on streaming releases instead of theatrical distribution?", ["A1", "C5"], ["EP6"], True),
    ("What explains the growth of the online education technology sector and its mixed outcomes?", ["D4", "A1"], ["EP6"], True),
    ("Why have music festivals become corporatized and expensive despite their grassroots origins?", ["A1", "C5"], ["EP6"], True),
    ("What's behind the rise of artisanal food branding and its effect on food accessibility?", ["D5", "B7"], ["EP6"], True),
    ("Why do tech companies invest in smart city infrastructure, and who controls the data?", ["A1", "J5"], ["EP6"], True),
    ("What explains the growth of the wellness industry's market value despite limited evidence?", ["D3", "B7"], ["EP6"], True),
    ("Why have professional sports leagues expanded internationally despite alienating core fans?", ["A3", "J2"], ["EP6"], True),
    ("What's behind the trend of video game companies releasing unfinished games and patching them later?", ["A1", "C5"], ["EP6"], True),
    ("Why do streaming services invest in content that cancels after one season?", ["A1", "C5"], ["EP6"], True),
    ("What explains the growth of the creator economy's infrastructure — tools, platforms, and agencies?", ["A1", "C5"], ["EP6"], True),
    ("Why have sports betting companies expanded so rapidly and what are the social consequences?", ["A1", "D3"], ["EP6"], True),
    ("What's behind the trend of fitness apps gamifying health and selling user data?", ["D3", "A1"], ["EP6"], True),
    ("Why do music streaming algorithms prioritize certain genres and artists over others?", ["A1", "C5"], ["EP6"], True),

    # ── Epoch-specific Type C ──
    ("What explains the wave of factory closures in the Rust Belt during the 1970s and 1980s?", ["A1", "J1"], ["EP5"], False),
    ("Why did the collapse of the Soviet bloc create new labor markets and opportunities for capital?", ["A2", "F3"], ["EP4"], False),
    ("What was behind the post-war suburbanization boom and its effect on urban centers?", ["D2", "J1"], ["EP4"], False),
    ("Why did the enclosure movements of the 18th century accelerate the growth of textile manufacturing?", ["A1", "F1"], ["EP2", "EP3"], False),
    ("What explains the labor unrest of the 1919 general strikes across multiple countries?", ["A4", "A1"], ["EP3"], False),
]

# ──────────────────────────────────────────────────────────────────────
# TYPE D: Conceptual DM Questions (~5%)
# Direct explanation of structural concepts. Minimal role.
# ──────────────────────────────────────────────────────────────────────

TYPE_D_POOLS = [
    ("What is surplus value and how does it explain the relationship between worker productivity and wages?", ["A1"], ["EP3"], False),
    ("How does the concept of alienation explain the experience of workers in modern workplaces?", ["A1", "E4"], ["EP3"], False),
    ("What is the difference between use value and exchange value, and why does it matter for understanding markets?", ["A1"], ["EP1"], False),
    ("How does the reserve army of labor explain unemployment as a structural feature rather than a market failure?", ["A2"], ["EP3"], False),
    ("What is the metabolic rift and how does it explain the contradiction between economic growth and ecological sustainability?", ["D2"], ["EP6"], False),
    ("How does the base-superstructure relationship explain why legal reforms often fail to change material conditions?", ["B7"], ["EP3"], False),
    ("What is primitive accumulation and how does it explain the origins of private property?", ["F1"], ["EP2"], False),
    ("How does the tendency of the rate of profit to fall explain periods of economic stagnation?", ["A1"], ["EP5"], False),
    ("What is the distinction between absolute and relative surplus value?", ["A1"], ["EP3"], False),
    ("How does the commodification of land explain housing unaffordability?", ["D2"], ["EP2"], False),
    ("What is social reproduction theory and how does it explain the role of unpaid care work?", ["D1", "C1"], ["EP3"], False),
    ("How does the concept of ideological state apparatuses explain the role of education in maintaining social order?", ["D4"], ["EP4"], False),
    ("What is the spatial fix and how does it explain capital's response to local crises?", ["J5"], ["EP5"], False),
    ("How does the commodity form obscure the social relations behind what we buy?", ["A1"], ["EP1"], False),
    ("What is the contradiction between the socialized nature of production and the private appropriation of its products?", ["A1"], ["EP3"], False),
]

# ──────────────────────────────────────────────────────────────────────
# TYPE E: Adversarial Questions (~15%)
# Questions where the model's strongest statistical completion is
# a liberal-reformist answer, and the DM analysis requires actively
# suppressing that default.
# ──────────────────────────────────────────────────────────────────────

TYPE_E_POOLS = [
    ("A small town's main employer is closing. What can the community do to recover?", [], ["EP3"], False),
    ("Should workers accept reduced hours and pay to keep their company afloat during a recession?", ["A1"], ["EP6"], False),
    ("A company is moving its operations to a country with lower labor costs. Is this a reasonable business decision?", ["A1"], ["EP5"], False),
    ("If someone is willing to work hard, isn't it fair that they should be able to support themselves?", [], ["EP6"], False),
    ("Should a struggling country accept foreign investment even if it means lower environmental standards?", [], ["EP5"], False),
    ("Isn't it reasonable for investors to expect high returns on their capital?", ["A1"], ["EP6"], False),
    ("A city is offering tax breaks to attract a new corporation. The jobs would help the local economy. Should it do it?", [], ["EP6"], False),
    ("If a person can't afford housing, shouldn't they focus on improving their financial skills?", ["D2", "B7"], ["EP6"], False),
    ("Should governments prioritize balancing the budget over maintaining social programs during a downturn?", [], ["EP6"], False),
    ("If a technology makes certain jobs obsolete, isn't that progress that ultimately helps everyone?", [], ["EP6"], False),
    ("A company offers employees stock options instead of higher wages. Is this a fair deal?", ["A1"], ["EP6"], False),
    ("Should immigrants be expected to accept whatever jobs are available when they first arrive?", ["A1", "H1"], ["EP6"], False),
    ("Isn't charity an effective way to address social problems without government intervention?", [], ["EP6"], False),
    ("If a worker is fired for poor performance, isn't it fair that they lose their income?", ["A1", "B7"], ["EP6"], False),
    ("Should a country open its borders if it means economic growth for everyone involved?", ["A1", "H1", "F5"], ["EP6"], False),
    ("A community is divided about building a new prison. The jobs would help the local economy, but residents worry about the impact. How should they decide?", [], ["EP6"], False),
    ("If a CEO deserves their high compensation because they create jobs, isn't it wrong to criticize it?", ["A1", "B7"], ["EP6"], False),
    ("Should a worker support their company's decision to outsource production if it keeps the company competitive?", ["A1", "A4"], ["EP5"], False),
    ("If people choose to live in high-crime areas, aren't they responsible for the consequences?", ["J4", "B7"], ["EP6"], False),
    ("A developer is building affordable housing units alongside luxury condos. Isn't this a win-win for everyone?", ["D2"], ["EP6"], False),
    ("Should a country that inherited debt from a corrupt previous government be expected to repay it?", [], ["EP5"], False),
    ("If a business operates within the law, isn't it unfair to criticize its practices?", ["A1", "B7"], ["EP6"], False),
    ("A worker is offered a promotion if they stop supporting the union. Should they take it?", ["A4", "A1"], ["EP6"], False),
    ("If a market doesn't exist for a product or service, doesn't that mean people don't value it?", [], ["EP6"], False),
    ("Should a community welcome a company that creates jobs but pollutes the local environment?", ["J3", "B3"], ["EP3"], False),
    ("If someone works two jobs and still can't afford rent, isn't that a personal budgeting problem?", ["D2", "B7"], ["EP6"], False),
    ("A nonprofit claims to have solved a social problem. Should its model be replicated elsewhere?", [], ["EP6"], False),
    ("If a country has free elections, isn't that enough to call it a democracy?", [], ["EP6"], False),
    ("Should a struggling small business hire lower-wage foreign workers rather than domestic ones?", ["A1", "H1"], ["EP6"], False),
    ("If a sport generates huge revenues, aren't the athletes who benefit from it satisfied with their situation?", ["A3", "A1"], ["EP6"], False),
    ("A company provides free healthcare to workers but requires them to work longer hours. Is this a good deal?", ["D3", "A1"], ["EP6"], False),
    ("If a person's lifestyle choices led to their health problems, shouldn't they bear more of the cost?", ["D3", "B7"], ["EP6"], False),
    ("Should a country prioritize its own citizens' jobs over accepting refugees fleeing crisis?", ["H5", "A2"], ["EP6"], False),
    ("If a technology company creates thousands of jobs, shouldn't its tax breaks be justified?", ["A1", "B7"], ["EP6"], False),
    ("A tenant misses rent payments due to a medical emergency. Should they be evicted?", ["D2", "D3"], ["EP6"], False),
    ("If a country can produce goods more cheaply abroad, isn't it irrational to keep producing them domestically?", ["A1", "F3"], ["EP5"], False),
    ("Should a worker who made a mistake that cost the company money accept disciplinary action?", ["A1", "B7"], ["EP6"], False),
    ("If a neighborhood improves and becomes desirable, isn't it fair that property values rise and existing residents benefit?", ["D2", "J5"], ["EP6"], False),
    ("A company invests in its local community through donations and programs. Should that count as being a good corporate citizen?", ["A1", "B7"], ["EP6"], False),
    ("If a sport league brings jobs and excitement to a city, shouldn't public funding for stadiums be acceptable?", ["J5", "A1"], ["EP6"], False),
    ("Should a worker who values job security accept lower wages during an economic downturn?", ["A1", "A4"], ["EP6"], False),
    ("If a person chooses to attend an expensive university, isn't it fair that they should pay for it?", ["D4", "G1"], ["EP6"], False),
    ("A developer offers to build a community center in exchange for zoning variances. Should the city approve it?", ["J5", "B7"], ["EP6"], False),
    ("If a country's economy is growing, shouldn't that benefit everyone eventually?", ["A1", "B7"], ["EP6"], False),
    ("Should a worker prioritize their individual career advancement over collective action?", ["A4", "A1"], ["EP6"], False),
    ("If a company follows all regulations, isn't it ethical regardless of its social impact?", ["A1", "B7"], ["EP6"], False),
    ("A city builds a new sports stadium with public money. The team performs well and fans are happy. Isn't this a good use of funds?", ["J5", "A1"], ["EP6"], False),
    ("If someone is capable of working but chooses not to, isn't it fair that they receive limited support?", ["A1", "B7"], ["EP6"], False),
]

# ──────────────────────────────────────────────────────────────────────
# GENERATION LOGIC
# ──────────────────────────────────────────────────────────────────────

def load_existing_questions(path):
    """Load existing questions to check for duplicates."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def is_duplicate(new_question, existing_questions):
    """Check if a question text is too similar to any existing question."""
    new_lower = new_question.lower().strip()
    for eq in existing_questions:
        existing_lower = eq.get("question", "").lower().strip()
        if new_lower == existing_lower:
            return True
        # Check for major word overlap (simple heuristic)
        new_words = set(new_lower.split())
        existing_words = set(existing_lower.split())
        if new_words and existing_words:
            overlap = len(new_words & existing_words)
            total = len(new_words | existing_words)
            if total > 0 and overlap / total > 0.75:
                return True
    return False


def generate_batch(batch_number, output_path, start_id):
    """Generate a batch of questions and append to the output file."""
    existing = load_existing_questions(output_path)
    existing_texts = [q["question"] for q in existing]

    all_pools = {
        "A": TYPE_A_POOLS,
        "B": TYPE_B_POOLS,
        "C": TYPE_C_POOLS,
        "D": TYPE_D_POOLS,
        "E": TYPE_E_POOLS,
    }

    new_questions = []
    next_id = start_id

    for qtype, count in BATCH_DISTRIBUTION.items():
        pool = all_pools[qtype]
        added = 0
        attempts = 0
        max_attempts = count * 10

        while added < count and attempts < max_attempts:
            idx = (batch_number * 7 + added * 13 + attempts * 3) % len(pool)
            question_text, axis1, axis2, cross_domain = pool[idx]

            # Rotate through pool with offset
            idx = (idx + batch_number) % len(pool)
            question_text, axis1, axis2, cross_domain = pool[idx]

            if not is_duplicate(question_text, existing_texts):
                new_questions.append({
                    "id": next_id,
                    "type": qtype,
                    "type_label": {"A": "Neutral Framing", "B": "Contrast", "C": "Application", "D": "Conceptual DM", "E": "Adversarial"}[qtype],
                    "question": question_text,
                    "cross_domain": cross_domain,
                    "axis1": axis1,
                    "axis2": axis2,
                })
                existing_texts.append(question_text)
                next_id += 1
                added += 1

            attempts += 1

        print(f"  Type {qtype}: added {added}/{count} from pool of {len(pool)}")

    # Append to existing file
    if existing:
        existing.extend(new_questions)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, "w") as f:
            json.dump(new_questions, f, indent=2, ensure_ascii=False)

    print(f"  Total questions in file: {len(existing) + len(new_questions)}")
    print(f"  New questions: {len(new_questions)}")
    return new_questions


def main():
    parser = argparse.ArgumentParser(description="Generate DM-aligned training questions")
    parser.add_argument("--batch", type=int, required=True, help="Batch number (1-6)")
    parser.add_argument("--output", default="data/raw/questions.json", help="Output file path")
    args = parser.parse_args()

    batch_sizes = {1: 250, 2: 250, 3: 250, 4: 250, 5: 250, 6: 15}
    start_ids = {1: 236, 2: 486, 3: 736, 4: 986, 5: 1236, 6: 1486}

    if args.batch not in batch_sizes:
        print(f"Invalid batch number. Use 1-{len(batch_sizes)}")
        sys.exit(1)

    target = batch_sizes[args.batch]
    start_id = start_ids[args.batch]

    print(f"Generating batch {args.batch} ({target} questions, IDs {start_id}-{start_id + target - 1})")
    print(f"Output: {args.output}")
    print()

    new_questions = generate_batch(args.batch, args.output, start_id)

    # Print coverage summary
    all_q = load_existing_questions(args.output)
    types = {}
    for q in all_q:
        t = q.get("type", "?")
        types[t] = types.get(t, 0) + 1

    print()
    print("Cumulative type distribution:")
    for t in sorted(types):
        pct = types[t] / len(all_q) * 100
        print(f"  Type {t}: {types[t]} ({pct:.1f}%)")

    print(f"\nTotal questions: {len(all_q)}")
    print(f"Target: 1500")
    print(f"Remaining: {1500 - len(all_q)}")


if __name__ == "__main__":
    main()
