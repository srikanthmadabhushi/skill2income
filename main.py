from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json, re, random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ENFORCEMENT RULES
# =========================

BANNED = ["freelance", "course", "blog", "tutorial"]
PLAN_KEYWORDS = ["30 day", "30 days", "plan", "roadmap", "week by week", "daily plan", "day by day"]
GENERIC_TITLE_PATTERNS = [
    "saas tool",
    "automation system",
    "template marketplace",
    "business platform",
    "service app",
]

ALLOWED_CATEGORIES = ["saas", "automation", "product", "marketplace", "consulting"]
FALLBACK_PATTERNS = {
    "SaaS": [
        ("Maintenance Cost Estimator", "Build a subscription app for teams that estimate maintenance, hosting, and upgrade costs before touching aging systems."),
        ("Compliance Change Tracker", "Create a niche tracking tool that alerts teams when internal rules, audit needs, or policy changes require updates."),
        ("Client Portal Builder", "Offer a self-serve platform that gives small agencies and IT firms branded client portals powered by your stack."),
    ],
    "Automation": [
        ("Lead Routing Engine", "Automate lead scoring, enrichment, and routing for small sales teams that waste time triaging inbound requests."),
        ("Invoice Exception Resolver", "Automate back-office workflows that flag broken invoices, missing fields, and payment mismatches before finance reviews them."),
        ("Support Triage Workflow", "Build workflow automations that classify support tickets, suggest fixes, and route issues to the right queue."),
    ],
    "Product": [
        ("Admin Panel Starter Kit", "Sell production-ready starter kits for operations dashboards, reporting portals, and internal CRUD systems."),
        ("Industry Template Pack", "Package reusable components, templates, and deployment blueprints for a specific niche with repeated workflow needs."),
        ("Migration Audit Toolkit", "Create a paid toolkit that helps teams assess legacy codebases, map risks, and plan staged rewrites."),
    ],
}


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def is_plan_request(text):
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in PLAN_KEYWORDS)


def get_user_messages(history):
    return [item.get("text", "") for item in history or [] if item.get("role") == "user" and item.get("text")]


def build_history_summary(history, limit=6):
    trimmed = []

    for item in history or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        role = item.get("role", "user")
        compact_text = " ".join(text.split())
        if len(compact_text) > 220:
            compact_text = compact_text[:220] + "..."
        trimmed.append(f"{role}: {compact_text}")

    return "\n".join(trimmed[-limit:])


def infer_focus(skills, history):
    cleaned_skills = (skills or "").strip()
    if cleaned_skills and not is_plan_request(cleaned_skills):
        return cleaned_skills

    user_messages = get_user_messages(history)

    for message in reversed(user_messages):
        if not is_plan_request(message):
            return message.strip()

    previous_titles, _ = extract_previous_ideas(history)
    if previous_titles:
        return next(iter(previous_titles))

    return cleaned_skills or "the selected skill"


def is_generic_title(title):
    normalized = normalize_text(title)
    return any(pattern in normalized for pattern in GENERIC_TITLE_PATTERNS)


def extract_previous_ideas(history):
    previous_titles = set()
    previous_models = set()

    for item in history or []:
        text = item.get("text", "")

        try:
            parsed = json.loads(text) if isinstance(text, str) else text
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            parsed = parsed.get("result", parsed.get("ideas", []))

        if isinstance(parsed, list):
            for idea in parsed:
                if not isinstance(idea, dict):
                    continue
                previous_titles.add(normalize_text(idea.get("title", "")))
                previous_models.add(normalize_text(idea.get("business_model", "")))

    return previous_titles, previous_models

def hard_filter(ideas):
    clean = []
    for idea in ideas:
        title = idea.get("title", "").lower()

        # ❌ remove bad patterns
        if any(b in title for b in BANNED):
            continue

        if is_generic_title(title):
            continue

        clean.append(idea)

    return clean


def enforce_categories(ideas):
    seen = set()
    final = []

    for idea in ideas:
        model = idea.get("business_model", "").lower()

        # map loosely into categories
        if "saas" in model:
            cat = "saas"
        elif "automation" in model:
            cat = "automation"
        elif "market" in model:
            cat = "marketplace"
        elif "product" in model or "template" in model:
            cat = "product"
        else:
            cat = "consulting"

        if cat not in seen:
            seen.add(cat)
            final.append(idea)

    return final


def remove_repeats(ideas, previous_titles):
    unique = []
    seen_titles = set(previous_titles)

    for idea in ideas:
        title_key = normalize_text(idea.get("title", ""))
        if not title_key or title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        unique.append(idea)

    return unique


def build_fallback_ideas(skills, previous_titles):
    ideas = []
    category_order = [("SaaS", 80, "Medium"), ("Automation", 75, "Low"), ("Product", 78, "Low")]

    for business_model, score, risk_level in category_order:
        options = FALLBACK_PATTERNS[business_model][:]
        random.shuffle(options)

        for suffix, description in options:
            title = f"{skills} {suffix}"
            if normalize_text(title) in previous_titles:
                continue

            ideas.append(
                {
                    "title": title,
                    "description": description,
                    "business_model": business_model,
                    "score": score,
                    "risk_level": risk_level,
                    "monthly_projection": {"month1":"$100","month2":"$700","month3":"$2200"}
                }
            )
            break

    return ideas


def is_valid(ideas):
    if len(ideas) < 3:
        return False

    titles = [i.get("title","").lower() for i in ideas]

    # ❌ if any banned slipped through
    if any(any(b in t for b in BANNED) for t in titles):
        return False

    if any(is_generic_title(t) for t in titles):
        return False

    # ❌ if all same type
    models = set(i.get("business_model","") for i in ideas)
    if len(models) < 2:
        return False

    return True


# =========================
# AI CALL
# =========================

def generate_ideas(skills, history_text, banned_titles, banned_models):
    banned_titles_text = ", ".join(sorted(banned_titles)) or "none"
    banned_models_text = ", ".join(sorted(banned_models)) or "none"

    prompt = f"""
You are a startup strategist.

Generate EXACTLY 3 income ideas.

Skills: {skills}
Context: {history_text}
Previously returned titles to avoid: {banned_titles_text}
Previously used business models to avoid reusing too heavily: {banned_models_text}

STRICT:
- Avoid freelancing, courses, blogs
- Focus on SaaS, automation, tools, marketplaces
- Ideas must belong to 3 different archetypes: one workflow SaaS, one automation service/product, one reusable product or toolkit
- Every idea must target a specific niche, audience, and pain point
- Do not repeat or lightly reword any previous title
- Avoid generic titles like "<skill> SaaS Tool", "<skill> Automation", or "<skill> Marketplace"
- Avoid broad audiences like "businesses", "everyone", or "developers" without a niche qualifier
- Make the concepts materially different from one another in customer, workflow, or monetization
- Use concrete names and descriptions, not generic placeholders
- Include these fields for each object: title, description, business_model, who_is_this_for, why_it_works, tools_needed, steps, time_to_start, earnings, monthly_estimate, difficulty, timeline, what_if, pro_tip, reason_for_best, personal_reason, score, risk_level, monthly_projection

Return JSON ARRAY.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        max_tokens=1400,
        messages=[
            {"role": "system", "content": "Generate high-quality startup ideas."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json","").replace("```","")

    try:
        return json.loads(txt)
    except:
        return []


def generate_plan(focus, history_text):
    prompt = f"""
You are a startup execution coach.

Create a practical 30-day execution plan for: {focus}

Conversation context:
{history_text}

STRICT:
- Return JSON ARRAY with EXACTLY 4 objects
- Each object must contain: week, goal, tasks
- week must be "Week 1", "Week 2", "Week 3", "Week 4"
- tasks must be an array of 6 to 8 concrete action items
- Build the plan as a real 30-day roadmap, not more income ideas
- Keep the plan specific to the focus area and previous chat context
- Do not use generic advice like "do research" without saying what to research
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": "Generate a practical 30-day action plan in JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json", "").replace("```", "")

    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return [
        {"week": "Week 1", "goal": f"Define the {focus} offer", "tasks": ["Pick one niche audience.", "List the main problem you will solve.", "Write the core offer promise.", "Choose a pricing hypothesis.", "Collect 5 examples of similar products.", "Draft a simple landing page outline."]},
        {"week": "Week 2", "goal": f"Build the first {focus} version", "tasks": ["Define the minimum feature set.", "Set up the project structure.", "Build the core workflow.", "Create one usable demo path.", "Test the main user flow end to end.", "Fix the most obvious friction points."]},
        {"week": "Week 3", "goal": "Validate with real users", "tasks": ["Share the demo with 5 target users.", "Collect objections and repeated questions.", "Improve the positioning message.", "Tighten the onboarding flow.", "Add one proof point or sample result.", "Prepare a short sales message."]},
        {"week": "Week 4", "goal": "Launch and iterate", "tasks": ["Publish the landing page.", "Reach out to 20 target users.", "Track replies and demo requests.", "Adjust pricing based on feedback.", "Document the onboarding steps.", "Plan the next month based on traction."]},
    ]


def generate_execution_plan(idea_title, idea_description, history_text):
    prompt = f"""
You are a startup execution coach.

Create a highly practical execution plan for this income idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

STRICT:
- Return JSON OBJECT
- Include these keys: title, summary, steps, quick_wins, first_offer
- title should be a short label for the plan
- summary should explain how to start in 2 sentences max
- steps must be an array of EXACTLY 5 concrete steps
- quick_wins must be an array of EXACTLY 3 fast actions
- first_offer must be one specific first offer the user can pitch or launch
- Do not return generic advice
- Keep actions specific to the idea title and description
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=900,
        messages=[
            {"role": "system", "content": "Generate a practical execution plan in JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json", "").replace("```", "")

    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "title": f"Launch Plan for {idea_title}",
        "summary": f"Start with one narrow customer segment for {idea_title} and build only the smallest usable version. Validate demand with real outreach before expanding features.",
        "steps": [
            "Pick one niche customer profile with a painful, repeated workflow.",
            "Outline the smallest version of the idea that solves one concrete problem.",
            "Build a clickable demo or simple working prototype for that core use case.",
            "Show the offer to 10 target users and collect objections and missing needs.",
            "Turn the best feedback into a paid pilot or early-access offer."
        ],
        "quick_wins": [
            "Write a one-sentence offer for the idea.",
            "Create a simple landing page or demo screen.",
            "Send outreach to 5 target prospects this week."
        ],
        "first_offer": f"Offer a paid pilot for {idea_title} to one niche customer group with setup support included."
    }


def generate_ai_coach_reply(idea_title, idea_description, user_message, history_text):
    prompt = f"""
You are an AI Income Coach.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation history:
{history_text}

Latest user message:
{user_message}

STRICT:
- Return JSON OBJECT
- Include these keys: reply, suggested_replies, next_steps
- reply should sound like a practical mentor, not a generic chatbot
- suggested_replies must be an array of EXACTLY 3 short follow-up options
- next_steps must be an array of EXACTLY 3 concrete actions
- Help the user refine, simplify, validate, or scale the selected idea
- Ask at most one direct question in the reply
- Keep the tone encouraging and specific
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=900,
        messages=[
            {"role": "system", "content": "You are a practical AI income coach who helps users refine and execute business ideas."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json", "").replace("```", "")

    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "reply": f"{idea_title} is a solid direction if you keep the first version narrow. Start with one niche customer, one painful workflow, and one offer that can be validated quickly. Do you want a beginner path or a faster-income path?",
        "suggested_replies": [
            "Give me a beginner version",
            "Give me a faster income version",
            "How do I find first customers?"
        ],
        "next_steps": [
            "Pick one niche audience for the idea.",
            "Define the smallest outcome your offer will deliver.",
            "Write a short pitch and show it to 5 target users."
        ]
    }


def generate_launch_agent_plan(idea_title, idea_description, history_text):
    prompt = f"""
You are a launch agent for early-stage income ideas.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

STRICT:
- Return JSON OBJECT
- Include these keys: niche, first_offer, landing_page_copy, outreach_message, seven_day_plan
- niche must be a short, specific target audience and pain point
- first_offer must be a concrete starter offer
- landing_page_copy must be an object with keys: headline, subheadline, cta
- outreach_message must be one ready-to-send outreach message
- seven_day_plan must be an array of EXACTLY 7 short actions, one per day
- Avoid generic startup advice
- Make the output specific to this idea
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a practical launch agent that turns ideas into immediate launch assets."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json", "").replace("```", "")

    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "niche": f"Small teams that need {idea_title} to solve one repeated workflow problem faster.",
        "first_offer": f"Offer a focused pilot of {idea_title} for one niche customer with setup and feedback support included.",
        "landing_page_copy": {
            "headline": f"Launch {idea_title} without the usual delay",
            "subheadline": "A focused solution for teams that need a simpler way to handle this workflow and get results quickly.",
            "cta": "Book an early access demo"
        },
        "outreach_message": f"Hi [Name], I am building {idea_title} for teams dealing with this workflow pain every week. I can show a simple pilot version and would love 15 minutes of feedback if this is relevant to your team.",
        "seven_day_plan": [
            "Day 1: Pick one narrow niche and define the one painful workflow you will solve.",
            "Day 2: Write the offer promise, target outcome, and starter pricing.",
            "Day 3: Create a basic landing page and short demo outline.",
            "Day 4: Build the smallest usable version or mockup.",
            "Day 5: Send outreach to 10 target users in the niche.",
            "Day 6: Collect objections, questions, and buying signals from replies.",
            "Day 7: Improve the offer and ask one strong prospect for a pilot."
        ]
    }


# =========================
# MAIN API
# =========================

@app.post("/generate-income-plan")
async def generate_income_plan(data: dict):

    skills = data.get("skills","")
    interests = data.get("interests", "")
    request_type = data.get("request_type", "").strip().lower()
    plan_focus = data.get("plan_focus", "").strip()
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history",[])
    history_text = build_history_summary(history)
    previous_titles, previous_models = extract_previous_ideas(history)
    focus = plan_focus or infer_focus(interests or skills, history)

    if request_type == "execution_plan":
        return {"result": generate_execution_plan(idea_title or focus, idea_description, history_text)}

    if request_type == "launch_agent":
        return {"result": generate_launch_agent_plan(idea_title or focus, idea_description, history_text)}

    if request_type == "plan" or is_plan_request(interests or skills):
        return {"result": generate_plan(focus, history_text)}

    max_attempts = 3
    attempt = 0
    ideas = []

    while attempt < max_attempts:

        raw = generate_ideas(focus, history_text, previous_titles, previous_models)

        # 🔥 ENFORCEMENT PIPELINE
        filtered = hard_filter(raw)
        filtered = remove_repeats(filtered, previous_titles)
        categorized = enforce_categories(filtered)

        ideas = categorized[:3]

        if is_valid(ideas):
            break

        attempt += 1

    # 🚨 FINAL FALLBACK (guaranteed output)
    if len(ideas) < 3:
        ideas = build_fallback_ideas(focus, previous_titles)

    # best option
    best = max(ideas, key=lambda x: x.get("score", 0))
    for idea in ideas:
        idea["best_option"] = (idea == best)

    return {"result": ideas}


@app.post("/ai-coach")
async def ai_coach(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)

    if not idea_title:
        return {
            "result": {
                "reply": "Choose an idea first, then I can help refine it into a practical path.",
                "suggested_replies": [
                    "Help me pick the best idea",
                    "Give me a beginner version",
                    "How do I validate demand?"
                ],
                "next_steps": [
                    "Pick one idea card to coach.",
                    "Share the version you like most.",
                    "Ask for refinement, validation, or growth."
                ]
            }
        }

    return {
        "result": generate_ai_coach_reply(
            idea_title=idea_title,
            idea_description=idea_description,
            user_message=user_message or "Help me move forward with this idea.",
            history_text=history_text
        )
    }


@app.post("/launch-agent")
async def launch_agent(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)

    return {
        "result": generate_launch_agent_plan(
            idea_title=idea_title or "the selected idea",
            idea_description=idea_description,
            history_text=history_text
        )
    }
