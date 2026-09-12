import json
import os
import random
import re
import urllib.request
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

try:
    from langchain_core.prompts import PromptTemplate
except Exception:  # pragma: no cover
    PromptTemplate = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

app = Flask(__name__)
CORS(app)


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, int(value)))


def extract_json_object(raw_text: str) -> Dict:
    if not raw_text:
        return {}
    text = raw_text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```\s*$', '', text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return __import__('json').loads(match.group(0))
        except Exception:
            return {}
    try:
        return __import__('json').loads(text)
    except Exception:
        return {}


def merge_ai_result(base: Dict, ai_data: Dict) -> Dict:
    if not ai_data:
        return base
    merged = dict(base)
    for key, value in ai_data.items():
        if value is not None:
            merged[key] = value
    return merged


def call_gemini_api(prompt: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.9,
            "topK": 40
        }
    }
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        candidates = data.get('candidates') or []
        if not candidates:
            return ""
        parts = candidates[0].get('content', {}).get('parts', [])
        texts = []
        for part in parts:
            if isinstance(part, dict) and 'text' in part:
                texts.append(part.get('text', ''))
        return '\n'.join(texts).strip()
    except Exception:
        return ""


def score_text(resume_text: str, job_role: str, job_description: str = "") -> Dict:
    text = (resume_text or "").strip()
    lower = text.lower()
    keywords = [
        "python", "javascript", "typescript", "react", "node", "sql", "excel", "api",
        "aws", "power bi", "communication", "problem solving", "leadership", "analytics",
        "product", "dashboard", "data", "machine learning", "cloud", "testing"
    ]
    role_keywords = []
    if job_role:
        role_keywords = [word.lower() for word in re.split(r"[\s/]+", job_role) if len(word) > 2]
    all_keywords = list(dict.fromkeys(role_keywords + keywords))
    matched = [kw for kw in all_keywords if kw in lower]
    missing = [kw for kw in all_keywords if kw not in lower]

    overall_score = 50 + (len(matched) * 4) + (5 if len(text.split()) > 200 else 0) + (5 if re.search(r"\d+", text) else 0)
    overall_score = clamp(overall_score)

    ats_score = clamp(overall_score - 5)
    skill_score = clamp(45 + (len(matched) * 5))
    keyword_score = clamp(40 + (len(matched) * 5))
    formatting_score = 85 if len(text) > 350 else 70 if len(text) > 180 else 55
    experience_score = 80 if any(word in lower for word in ["experience", "worked", "developed", "built"]) else 60
    education_score = 80 if any(word in lower for word in ["b.tech", "bachelor", "education", "degree", "masters"]) else 65
    achievement_score = 80 if any(word in lower for word in ["project", "achievement", "certification", "award"]) else 60
    job_match_score = clamp((skill_score + keyword_score) // 2)

    ats_verdict = "Excellent" if ats_score >= 85 else "Good" if ats_score >= 65 else "Needs Improvement"

    strengths = [
        {
            "title": "Relevant technical foundation",
            "detail": f"{len(matched)} skills and role keywords appear to align with the {job_role or 'target'} profile."
        },
        {
            "title": "Professional structure",
            "detail": "The resume contains a readable structure and enough content to be interpreted by ATS systems."
        },
        {
            "title": "Action-oriented evidence",
            "detail": "Resume text includes performance and project language that signals measurable contribution."
        }
    ]

    weaknesses = [
        {
            "title": "Keyword gap",
            "detail": f"A few company-critical terms are still missing: {', '.join(missing[:5]) if missing else 'none detected'}.",
            "fix": "Add the missing keywords naturally into relevant skills, projects, and experience bullet points."
        },
        {
            "title": "Need stronger metrics",
            "detail": "Recruiters favor quantified results, so adding numbers like percentage, revenue, users, or time saved will improve credibility.",
            "fix": "Turn tasks into achievement bullets using results, scope, and measurable outcomes."
        },
        {
            "title": "Role alignment gap",
            "detail": "Your resume can be made more aligned with the exact job description by matching the wording used by the employer.",
            "fix": "Mirror the job description language in your summary, skills, and project descriptions."
        }
    ]

    section_analysis = {
        "summary": {"present": True, "score": 72, "note": "Summary is reasonably present and can be sharpened for stronger role alignment."},
        "experience": {"present": True, "score": experience_score, "note": "The experience section is readable and can be made more result-driven."},
        "skills": {"present": True, "score": skill_score, "note": "The skill section is a good starting point; add missing role keywords where genuine."},
        "education": {"present": True, "score": education_score, "note": "Education appears present and ATS-friendly."},
        "projects": {"present": True, "score": achievement_score, "note": "Projects help strengthen the application and should highlight impact."}
    }

    company_fit = {
        "company": {"name": (job_role or "Target Role") + " profile"},
        "fitScore": clamp(50 + (len(matched) * 3)),
        "shortlistStatus": "Strong shortlist fit" if clamp(50 + (len(matched) * 3)) >= 70 else "Moderate shortlist fit" if clamp(50 + (len(matched) * 3)) >= 50 else "Needs more alignment",
        "criteria": [
            {"keyword": kw, "matched": kw in lower}
            for kw in all_keywords[:12]
        ]
    }

    selection_probability = clamp(50 + (overall_score * 0.35) + (company_fit["fitScore"] * 0.2))

    result = {
        "overallScore": overall_score,
        "atsScore": ats_score,
        "skillScore": skill_score,
        "keywordScore": keyword_score,
        "formattingScore": formatting_score,
        "experienceScore": experience_score,
        "educationScore": education_score,
        "achievementScore": achievement_score,
        "jobMatchScore": job_match_score,
        "atsVerdict": ats_verdict,
        "matchingKeywords": matched[:12],
        "missingKeywords": missing[:12],
        "recommendedKeywords": missing[:6],
        "matchingSkills": matched[:10],
        "missingSkills": missing[:10],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": [
            {
                "section": "Summary",
                "current": "Resume summary can be more job-specific.",
                "improved": f"Experienced {job_role or 'professional'} with skills in {', '.join(matched[:4]) if matched else 'core business and technical capabilities'} delivering measurable value.",
                "why": "A targeted summary helps recruiters and ATS systems map the resume correctly."
            }
        ],
        "sectionAnalysis": section_analysis,
        "companyFit": company_fit,
        "selectionProbability": selection_probability,
        "salaryRange": {"label": "Not estimated"},
        "jobRole": job_role,
        "resumeFileName": "resume.txt",
        "createdAt": __import__('datetime').datetime.utcnow().isoformat(),
        "source": "langchain-backend"
    }
    return result


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    payload = request.get_json(silent=True) or {}
    resume_text = payload.get('resumeText', '')
    job_role = payload.get('jobRole', '')
    job_description = payload.get('jobDescription', '')

    result = score_text(resume_text, job_role, job_description)

    gemini_prompt = (
        "You are an ATS and hiring expert. Analyze this resume against the job role and job description provided. "
        "Focus on the user's real strengths, missing skills, weak points, ATS issues, and relevant company-fit gaps. "
        "Return ONLY valid JSON with the following keys: "
        "overallScore, atsVerdict, strengths, weaknesses, missingSkills, matchingSkills, selectionProbability, companyFit, raw_analysis. "
        "Each item in strengths and weaknesses must have title and detail. "
        "missingSkills and matchingSkills must be arrays of strings. "
        "selectionProbability must be an integer between 0 and 100. "
        "companyFit must include company name and fitScore. "
        "Solve the user's actual resume problem: identify what is good, what is lacking, and what is missing for the target role.\n\n"
        f"Job role: {job_role or 'target role'}\nJob description: {job_description or 'No job description provided.'}\nResume:\n{resume_text}"
    )

    gemini_reply = call_gemini_api(gemini_prompt, temperature=0.2, max_tokens=1200)
    if gemini_reply:
        try:
            ai_data = extract_json_object(gemini_reply)
            if ai_data:
                result = merge_ai_result(result, ai_data)
            result["raw_analysis"] = gemini_reply
        except Exception:
            result["raw_analysis"] = gemini_reply
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and ChatOpenAI and PromptTemplate:
            try:
                llm = ChatOpenAI(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    temperature=0.2,
                    api_key=api_key
                )
                prompt = PromptTemplate.from_template(
                    "You are an ATS and hiring expert. Analyze this resume against the job role and job description provided. "
                    "Focus on the user's real strengths, missing skills, weak points, ATS issues, and relevant company-fit gaps. "
                    "Return ONLY valid JSON with the following keys: "
                    "overallScore, atsVerdict, strengths, weaknesses, missingSkills, matchingSkills, selectionProbability, companyFit, raw_analysis. "
                    "Each item in strengths and weaknesses must have title and detail. "
                    "missingSkills and matchingSkills must be arrays of strings. "
                    "selectionProbability must be an integer between 0 and 100. "
                    "companyFit must include company name and fitScore. "
                    "Solve the user's actual resume problem: identify what is good, what is lacking, and what is missing for the target role.\n\n"
                    "Job role: {role}\nJob description: {jd}\nResume:\n{text}"
                )
                chain = prompt | llm
                response = chain.invoke({
                    "role": job_role or "target role",
                    "jd": job_description or "No job description provided.",
                    "text": resume_text
                })
                raw_text = getattr(response, 'content', str(response))
                ai_data = extract_json_object(raw_text)
                if ai_data:
                    result = merge_ai_result(result, ai_data)
                result["raw_analysis"] = raw_text
            except Exception as e:  # pragma: no cover
                result["raw_analysis"] = f"LLM unavailable: {str(e)}"
        else:
            result["raw_analysis"] = "No Gemini or OpenAI API key supplied. Using local resume heuristic analysis."

    return jsonify(result)


def generate_chat_reply(question: str) -> str:
    q = (question or '').strip().lower()
    if not q:
        return "Ask me anything about resumes, ATS, interviews, job roles, or skill gaps. I can help you improve your profile."

    role_hint = ""
    role_terms = [
        "data analyst", "software engineer", "product manager", "frontend developer",
        "backend developer", "full stack developer", "ui/ux designer", "data scientist",
        "marketing manager", "sales executive", "project manager", "business analyst"
    ]
    for term in role_terms:
        if term in q:
            role_hint = term
            break

    def pick(items):
        return random.choice(items)

    if any(k in q for k in ['resume', 'ats', 'cv', 'job application', 'score']):
        variants = [
            f"For a strong resume, keep the format simple, match the wording of the job description, and add measurable wins. The fastest upgrade is usually a tighter summary and results-focused bullets for {role_hint or 'your target role'}.",
            "A recruiter-friendly resume is clear, keyword-aligned, and outcome-driven. Remove vague responsibilities and replace them with concrete impact, numbers, and role-specific proof.",
            "The best ATS resumes feel easy to scan. Use standard headings, mirror the job language naturally, and let your experience show what changed because of your work."
        ]
        return pick(variants)
    if any(k in q for k in ['salary', 'pay', 'package', 'compensation', 'lpa']):
        variants = [
            "Salary usually depends on experience level, role, location, and how strongly your resume proves value. The stronger your evidence, the stronger your negotiation position.",
            "A realistic pay range comes from market data plus proof of impact. If your resume shows measurable outcomes and the right keywords, you usually have more leverage in salary talks.",
            "The right compensation story is built from scope, seniority, and results. Make your profile read like a high-impact candidate and the numbers become easier to justify."
        ]
        return pick(variants)
    if any(k in q for k in ['skill', 'missing', 'gap', 'learn', 'improve', 'weak']):
        variants = [
            "Target the top 2–3 skills from the job description and build proof around them. A focused improvement plan is stronger than listing every possible keyword you do not actually use.",
            "Improve the skills that matter most to the role, then show them through examples, projects, and clear impact. That is how you turn a gap into a credible strength.",
            "The most effective skill upgrade is practical: learn the core requirement, apply it in a project, and then reflect that work clearly in your resume and interview stories."
        ]
        return pick(variants)
    if any(k in q for k in ['interview', 'prepare', 'mock', 'job', 'offer']):
        variants = [
            "Prepare three stories: leadership, problem-solving, and teamwork. Keep each answer structured around the challenge, your actions, and the measurable outcome.",
            "Strong interview answers are concrete and brief. Explain the business problem, what you did, and why it mattered in terms of time, revenue, quality, or customer value.",
            "Use STAR, but make it sound natural. Recruiters care less about the format and more about whether your examples show judgment, ownership, and impact."
        ]
        return pick(variants)
    if any(k in q for k in ['linkedin', 'portfolio', 'github', 'profile']):
        variants = [
            "Keep your LinkedIn, GitHub, and portfolio aligned with your resume. Use the same role keywords and emphasize outcomes, not just tools or responsibilities.",
            "Your online profile should tell the same story as your resume: the work you do, the value you create, and the skills you want to be hired for.",
            "Consistency matters. If your resume says you lead analytics and your GitHub says nothing about results, the recruiter will feel the mismatch immediately."
        ]
        return pick(variants)
    if any(k in q for k in ['hello', 'hi', 'hey', 'how are you', 'good morning', 'good evening']):
        variants = [
            "Hello! I can help you sharpen your resume, improve ATS fit, and plan the next best move for your target role.",
            "Hi there — I can review your resume, identify what is missing, and suggest practical improvements based on the role you want.",
            "Welcome! Tell me the role, your experience, and the gap you want to improve, and I will help you plan a stronger application."
        ]
        return pick(variants)

    variants = [
        f"Start with the target role and map your experience to the problems the employer is trying to solve. Then turn your strongest examples into clear, outcome-based language for {role_hint or 'your profile'}.",
        "The strongest answer usually combines role-specific tailoring with proof. Show what you built, what changed, and how it connects to the job requirement.",
        "A useful strategy is to anchor your response in the job description, highlight the most relevant wins, and remove anything that dilutes your fit with the role."
    ]
    return pick(variants)


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    payload = request.get_json(silent=True) or {}
    question = payload.get('question', '')
    reply = generate_chat_reply(question)

    gemini_reply = call_gemini_api(
        f"You are a helpful AI career coach for resume, ATS, and hiring questions. Answer the user clearly and practically, in 2–5 paragraphs max. Keep it concise but useful.\n\nUser question: {question}",
        temperature=0.5,
        max_tokens=600
    )
    if gemini_reply:
        reply = gemini_reply.strip()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and ChatOpenAI and PromptTemplate:
            try:
                llm = ChatOpenAI(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    temperature=0.3,
                    api_key=api_key
                )
                prompt = PromptTemplate.from_template(
                    "You are a helpful AI career coach for resume, ATS, and hiring questions. "
                    "Answer the user clearly and practically, in 2–5 paragraphs max. "
                    "Keep it concise but useful.\n\nUser question: {question}"
                )
                chain = prompt | llm
                response = chain.invoke({"question": question})
                text = getattr(response, 'content', str(response))
                if isinstance(text, str) and text.strip():
                    reply = text.strip()
            except Exception:
                pass

    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
