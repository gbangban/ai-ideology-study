import json
from collections import Counter

data = json.load(open('data/raw/questions.json'))

# Current cross-domain distribution
cross_qs = [q for q in data if q["cross_domain"]]
current_cats = Counter()
for q in cross_qs:
    for a in q["axis1"]:
        if a != "cross":
            current_cats[a[0]] += 1

TARGETS = {
    'A': 40, 'B': 35, 'C': 30, 'D': 20, 'E': 15,
    'F': 25, 'G': 20, 'H': 20, 'I': 25, 'J': 20, 'K': 30,
}

print(f"Current: {dict(current_cats)}")
print(f"Target: {TARGETS}")

# Calculate what needs to be added
to_add = {}
for cat in TARGETS:
    diff = TARGETS[cat] - current_cats.get(cat, 0)
    if diff > 0:
        to_add[cat] = diff

print(f"\nTo add: {to_add}")
print(f"Total to add: {sum(to_add.values())}")

def q(id, typ, qtext, cross_domain=False, axis1=None, axis2=None):
    return {
        "id": id, "type": typ,
        "type_label": {"A": "Neutral Framing", "B": "Contrast", "C": "Application", "D": "Conceptual DM", "E": "Adversarial"}[typ],
        "question": qtext, "cross_domain": cross_domain,
        "axis1": axis1 or [], "axis2": axis2 or []
    }

# New questions with varied structures
new_questions = []

# B (Race) - 13 questions
new_questions.extend([
    q(0, "C", "A hiring algorithm flags resumes from certain neighborhoods as lower quality. What forces shape whose applications get filtered out?", True, ["B3", "cross"], ["EP6"]),
    q(0, "B", "A neighborhood's property values rise as new amenities arrive, but long-term residents can no longer afford to stay. What emerges when community improvement means community displacement?", True, ["B4", "J1", "cross"], ["EP6"]),
    q(0, "A", "A school district's funding depends on local property taxes. How might we understand educational inequality when wealth and race map onto the same zip codes?", True, ["B1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A restaurant chain uses traditional designs from a culture it does not belong to. What does this practice reveal about who benefits from cultural borrowing?", True, ["B3", "cross"], ["EP6"]),
    q(0, "A", "A city's police budget receives more funding than its mental health services. What structural factors explain this allocation choice?", True, ["B1", "cross"], ["EP6"]),
    q(0, "B", "A company's diversity report highlights demographic numbers but does not address promotion patterns. How might we understand the difference between representation and power?", True, ["B3", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's air quality deteriorates near industrial zones. What conditions enable certain communities to bear environmental hazards others avoid?", True, ["B1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A social media platform removes content from certain regions while leaving similar content from others untouched. What does this selective enforcement reveal about power?", True, ["B3", "cross"], ["EP6"]),
    q(0, "A", "A university's legacy admission preference favors applicants from certain families. How might this practice reproduce racial and class hierarchies across generations?", True, ["B1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A film studio casts white actors in roles written as characters of color. What does this practice tell us about who controls cultural representation?", True, ["B3", "cross"], ["EP6"]),
    q(0, "A", "A town holds a referendum on building affordable housing. What structural barriers prevent residents from voting for projects that would help their neighbors?", True, ["B1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A healthcare study excludes participants who cannot read English. What does this methodological choice do to the validity of the findings for multilingual communities?", True, ["B3", "D1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's tree canopy coverage correlates with income levels. What conditions produce this environmental inequality?", True, ["B1", "J1", "cross"], ["EP6"]),
])

# C (Gender) - 12 questions
new_questions.extend([
    q(0, "C", "A company's parental leave policy gives more weeks to birthing parents than to adopting parents. What does this distinction reveal about how workplaces value different kinds of care?", True, ["C1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A sports broadcast focuses on female athletes' outfits rather than their performance. How might we understand the relationship between visibility and objectification in media coverage?", True, ["C1", "cross"], ["EP6"]),
    q(0, "A", "A city's public spaces lack nursing rooms and accessible playgrounds. What conditions enable urban design to assume a male default user?", True, ["C1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A tech company's product team designs a pregnancy app without consulting the people who will use it. What does this design process reveal about who gets to define needs?", True, ["C1", "cross"], ["EP6"]),
    q(0, "A", "A workplace's networking events happen after hours at venues that exclude people with caregiving responsibilities. How might we understand the relationship between informal networks and career advancement?", True, ["C1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A fashion brand releases a 'maternity line' that prices pregnancy clothing at a premium. What does this market segment tell us about how reproduction is commodified?", True, ["C1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A university's title IX office handles complaints through a process that mirrors criminal courts. How might we understand the relationship between institutional power and survivor agency?", True, ["C1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's domestic violence shelter faces zoning opposition. What conditions enable communities to resist services that protect vulnerable residents?", True, ["C1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A streaming series portrays a female lead as 'strong' while reducing her to a plot device for male character development. What does this representation reveal about the limits of feminist media?", True, ["C1", "cross"], ["EP6"]),
    q(0, "A", "A company's performance reviews use subjective language that penalizes women for assertiveness. How might we understand the relationship between evaluation systems and gendered behavior?", True, ["C1", "cross"], ["EP6"]),
    q(0, "C", "A social media algorithm promotes content that reinforces traditional gender roles. What does this pattern tell us about the relationship between engagement metrics and social norms?", True, ["C1", "cross"], ["EP6"]),
    q(0, "B", "A city's park redesign removes a community garden that women organized. How might we understand the relationship between public space design and informal labor?", True, ["C1", "J1", "cross"], ["EP6"]),
])

# E (Disability) - 1 question
new_questions.extend([
    q(0, "A", "A city's public buildings are fully accessible but the surrounding sidewalks are not. How might we understand the relationship between built environment and mobility?", True, ["E1", "J1", "cross"], ["EP6"]),
])

# F (Coloniality/Indigeneity) - 4 questions
new_questions.extend([
    q(0, "C", "A mining company extracts rare earth minerals for green technology from indigenous land. What does this contradiction reveal about the relationship between environmental transition and colonial extraction?", True, ["F3", "cross"], ["EP6"]),
    q(0, "B", "A museum displays sacred objects collected during colonial expeditions. How might we understand the relationship between cultural preservation and cultural theft?", True, ["F3", "cross"], ["EP6"]),
    q(0, "A", "A city's name honors a settler who signed treaties that displaced the original inhabitants. What does the debate over renaming reveal about whose history gets commemorated?", True, ["F2", "J1", "cross"], ["EP6"]),
    q(0, "B", "A pharmaceutical company patents a plant used for centuries by indigenous healers. What does this intellectual property claim construct about knowledge ownership?", True, ["F3", "cross"], ["EP6"]),
])

# G (Age/Generational) - 6 questions
new_questions.extend([
    q(0, "C", "A pension fund shifts from defined-benefit to defined-contribution plans. How might we understand the relationship between retirement security and intergenerational wealth transfer?", True, ["G3", "cross"], ["EP6"]),
    q(0, "B", "A workplace's age discrimination complaint process requires evidence that older workers cannot learn new technology. What does this burden of proof construct about aging and competence?", True, ["G3", "cross"], ["EP6"]),
    q(0, "A", "A city's housing policy favors young professional buyers over multi-generational families. How might we understand the relationship between housing markets and family structure?", True, ["G3", "J1", "cross"], ["EP6"]),
    q(0, "B", "A healthcare system prioritizes treatments for younger patients over palliative care for the elderly. How might we understand the relationship between medical value and age?", True, ["G3", "D1", "cross"], ["EP6"]),
    q(0, "A", "A school district cuts arts and music programs first. How might we understand the relationship between educational priorities and the life stages of students?", True, ["G3", "D1", "cross"], ["EP6"]),
    q(0, "C", "A tech company's workforce skews younger as older workers are displaced by AI. What does this demographic shift reveal about the relationship between technological change and age?", True, ["G3", "cross"], ["EP6"]),
])

# H (Immigration) - 4 questions
new_questions.extend([
    q(0, "C", "A hospital employs migrant workers who cannot access the same benefits as citizen staff. How might we understand the relationship between documentation status and workplace rights?", True, ["H1", "cross"], ["EP6"]),
    q(0, "B", "A border wall construction project displaces indigenous communities who do not recognize national boundaries. How might we understand the relationship between border enforcement and indigenous sovereignty?", True, ["H1", "F3", "cross"], ["EP6"]),
    q(0, "A", "A city's immigrant population grows but public services do not expand accordingly. What conditions enable this service gap?", True, ["H1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A company sponsors work visas for skilled workers while opposing immigration reform for low-wage workers. How might we understand the relationship between labor demand and immigration policy?", True, ["H1", "A1", "cross"], ["EP6"]),
])

# I (Religion/Secularism) - 20 questions
new_questions.extend([
    q(0, "C", "A workplace's holiday calendar only recognizes Christian holidays. How might we understand the relationship between institutional time and religious default?", True, ["I1", "cross"], ["EP6"]),
    q(0, "B", "A school's prayer space is available but no equivalent accommodation exists for secular meditation. What does this asymmetry construct about religious privilege?", True, ["I1", "cross"], ["EP6"]),
    q(0, "A", "A religious institution owns housing that it rents below market rate to congregants. How might we understand the relationship between faith communities and housing provision?", True, ["I1", "J1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A city's zoning board denies a permit for a mosque while approving a church expansion. How might we understand the relationship between religious freedom and local governance?", True, ["I1", "cross"], ["EP6"]),
    q(0, "A", "A workplace requires employees to participate in voluntary team-building at a church. How might we understand the relationship between social cohesion and religious coercion?", True, ["I1", "cross"], ["EP6"]),
    q(0, "C", "A religious organization runs food banks that require prayer participation. How might we understand the relationship between charitable aid and religious conversion?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A school curriculum teaches about world religions from a secular perspective but allows religious groups to teach their own faiths. How might we understand the relationship between secular education and religious instruction?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's religious diversity increases as immigrant populations grow. How might we understand the relationship between demographic change and religious landscape?", True, ["I1", "H1", "cross"], ["EP6"]),
    q(0, "B", "A charity's disaster relief prioritizes communities with religious institutions that can coordinate distribution. How might we understand the relationship between faith networks and aid access?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "A", "A city's public funding goes to faith-based social services while secular equivalents receive less. How might we understand the relationship between state support and religious organization?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A religious school receives public vouchers while secular schools do not. How might we understand the relationship between educational choice and religious subsidy?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A workplace's anti-discrimination policy covers religion but not religious practice. How might we understand the relationship between belief and observance in legal protection?", True, ["I1", "cross"], ["EP6"]),
    q(0, "A", "A community's religious institutions serve as gathering spaces during isolation. How might we understand the relationship between faith communities and social infrastructure?", True, ["I1", "cross"], ["EP6"]),
    q(0, "B", "A city's historical markers commemorate religious founders but not indigenous spiritual sites. How might we understand the relationship between public memory and religious dominance?", True, ["I1", "F3", "cross"], ["EP6"]),
    q(0, "C", "A city's public broadcasting allocates more airtime to religious programming than to secular community events. How might we understand the relationship between media access and religious visibility?", True, ["I1", "cross"], ["EP6"]),
    q(0, "B", "A workplace's dress code prohibits religious symbols. How might we understand the relationship between professional norms and religious expression?", True, ["I1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's religious diversity increases as immigrant populations grow. How might we understand the relationship between demographic change and social cohesion?", True, ["I1", "H1", "cross"], ["EP6"]),
    q(0, "B", "A school's parent-teacher organization meets during work hours, excluding working parents. How might we understand the relationship between school governance and class-based participation?", True, ["I1", "D1", "cross"], ["EP6"]),
    q(0, "A", "A city's public parks host religious gatherings without equivalent secular programming. How might we understand the relationship between public space use and religious privilege?", True, ["I1", "J1", "cross"], ["EP6"]),
    q(0, "C", "A charity's disaster relief prioritizes communities with religious institutions that can coordinate distribution. How might we understand the relationship between faith networks and aid access?", True, ["I1", "D1", "cross"], ["EP6"]),
])

# K (Intersectional) - 29 questions
new_questions.extend([
    q(0, "C", "A workplace's diversity initiative focuses on gender without addressing how race and class shape different women's experiences. How might we understand the relationship between single-axis and intersectional analysis?", True, ["K1", "cross"], ["EP6"]),
    q(0, "B", "A policy designed to help 'vulnerable populations' treats all poor people the same. How might we understand the relationship between universal design and differential impact?", True, ["K1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's elderly residents face isolation compounded by language barriers and disability. How might we understand the relationship between age, migration, and social connection?", True, ["K3", "G3", "cross"], ["EP6"]),
    q(0, "B", "A healthcare study's inclusion criteria exclude people with disabilities. How might we understand the relationship between research design and population representation?", True, ["K1", "E1", "cross"], ["EP6"]),
    q(0, "A", "A city's affordable housing lottery does not account for families with multiple disabilities. How might we understand the relationship between housing policy and intersectional need?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A social media algorithm promotes content that reinforces stereotypes about certain communities. How might we understand the relationship between algorithmic bias and intersectional identity?", True, ["K1", "cross"], ["EP6"]),
    q(0, "B", "A workplace's harassment policy addresses gender and race separately but not their combination. How might we understand the relationship between legal categories and lived experience?", True, ["K1", "cross"], ["EP6"]),
    q(0, "A", "A community's domestic violence services do not accommodate religious practices. How might we understand the relationship between support services and cultural accessibility?", True, ["K1", "I1", "cross"], ["EP6"]),
    q(0, "B", "A political campaign's messaging targets 'working families' without addressing how race and gender shape different workers' experiences. How might we understand the relationship between political framing and intersectional reality?", True, ["K1", "cross"], ["EP6"]),
    q(0, "A", "A city's emergency preparedness plan does not account for people who cannot evacuate due to disability or language barriers. How might we understand the relationship between crisis planning and intersectional vulnerability?", True, ["K1", "E1", "cross"], ["EP6"]),
    q(0, "C", "A university's financial aid package does not consider the additional costs of disability accommodations. How might we understand the relationship between educational access and intersectional need?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A city's public transit route map shows connections between wealthy neighborhoods but not between working-class areas. How might we understand the relationship between transit planning and class geography?", True, ["K1", "J1", "cross"], ["EP6"]),
    q(0, "A", "A school's gifted program enrollment skews toward wealthy, white students. How might we understand the relationship between identification criteria and demographic representation?", True, ["K1", "B1", "cross"], ["EP6"]),
    q(0, "B", "A workplace's flexible work policy requires employees to be available during core business hours. How might we understand the relationship between flexibility and caregiving responsibilities?", True, ["K1", "C1", "cross"], ["EP6"]),
    q(0, "C", "A city's green gentrification projects improve environmental quality but displace long-term residents. How might we understand the relationship between environmental justice and racial equity?", True, ["K1", "B1", "cross"], ["EP6"]),
    q(0, "B", "A healthcare system's telehealth options exclude patients without reliable internet access. How might we understand the relationship between digital access and health equity?", True, ["K1", "E1", "cross"], ["EP6"]),
    q(0, "A", "A neighborhood's food desert overlaps with a high pollution zone. How might we understand the relationship between environmental hazard and food access?", True, ["K1", "J1", "cross"], ["EP6"]),
    q(0, "B", "A university's financial aid package does not consider the additional costs of disability accommodations. How might we understand the relationship between educational access and intersectional need?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A city's emergency shelter hours conflict with shift work schedules. How might we understand the relationship between social services and work time?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A workplace's wellness program offers yoga classes during lunch but not childcare during those same hours. How might we understand the relationship between employee benefits and care responsibilities?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "A", "A city's public transportation schedule does not align with shift work patterns. How might we understand the relationship between transit design and care worker mobility?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A family's time poverty increases as both parents work multiple jobs. How might we understand the relationship between income generation and care capacity?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A school's parent-teacher conferences are scheduled during work hours. How might we understand the relationship between school-family engagement and work flexibility?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A city's senior center offers activities that assume participants have transportation. How might we understand the relationship between care infrastructure and mobility access?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "A", "A workplace's mental health days require advance notice, excluding acute stress. How might we understand the relationship between care policies and care needs?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A city's community center programming reflects middle-class leisure preferences. How might we understand the relationship between public recreation and class-based time use?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "C", "A family's care burden increases when a relative develops a chronic illness. How might we understand the relationship between health outcomes and care capacity?", True, ["K1", "D1", "cross"], ["EP6"]),
    q(0, "B", "A city's emergency alert system only provides audio warnings. How might we understand the relationship between crisis communication and sensory access?", True, ["K1", "E1", "cross"], ["EP6"]),
    q(0, "A", "A workplace's team-building activity requires physical participation. How might we understand the relationship between social cohesion and bodily capacity?", True, ["K1", "E1", "cross"], ["EP6"]),
])

# Add new questions to data
for item in new_questions:
    data.append(item)

# Renumber IDs
data.sort(key=lambda x: x["id"])
for i, q_item in enumerate(data):
    q_item["id"] = i + 1

# Verify
cross_qs_final = [q for q in data if q["cross_domain"]]
final_cats = Counter()
for q in cross_qs_final:
    for a in q["axis1"]:
        if a != "cross":
            final_cats[a[0]] += 1

print(f"\nFinal cross-domain categories: {dict(final_cats)}")
print(f"Final cross-domain total: {len(cross_qs_final)}")
print(f"Total questions: {len(data)}")

# Check for duplicate questions
questions = [q["question"] for q in data]
unique_questions = set(questions)
print(f"\nUnique questions: {len(unique_questions)} / {len(questions)}")

# Check schema
errors = 0
for q_item in data:
    for r in ["id", "type", "type_label", "question", "cross_domain", "axis1", "axis2"]:
        if r not in q_item:
            errors += 1
    if not isinstance(q_item.get("cross_domain"), bool):
        errors += 1

print(f"Schema errors: {errors}")

# Write
json.dump(data, open('data/raw/questions.json', 'w'), indent=2, ensure_ascii=False)
print("\nWritten to data/raw/questions.json")
