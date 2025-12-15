# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import spacy
import fitz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from spacy.matcher import Matcher, PhraseMatcher
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from fpdf import FPDF
from docx import Document

# Define classes first
class EnhancedResumeParser:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns()
    
    def _add_patterns(self):
        self.matcher.add("EDUCATION", [
            [{"LOWER": {"IN": ["bachelor", "master", "phd", "bs", "ms"]}}],
            [{"LOWER": "degree"}, {"LOWER": "in"}]
        ])
        
        self.matcher.add("EXPERIENCE", [
            [{"ENT_TYPE": "DATE"}, {"LOWER": {"IN": ["years", "yrs"]}}]
        ])
    
    def extract_experience(self, text):
        doc = self.nlp(text)
        experiences = re.findall(r'(\d+)\+? (?:years?|yrs?)', text, re.I)
        return f"{max(experiences, default=0)} years" if experiences else "Not specified"
    
    def extract_education(self, text):
        doc = self.nlp(text)
        degrees = []
        for match_id, start, end in self.matcher(doc):
            if self.matcher.vocab.strings[match_id] == "EDUCATION":
                degrees.append(doc[start:end].text)
        return degrees if degrees else ["Education not found"]
    
    def extract_location(self, text):
        doc = self.nlp(text)
        locations = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                try:
                    loc = geolocator.geocode(ent.text, exactly_one=True)
                    if loc:
                        locations.append(f"{ent.text} ({loc.latitude:.4f}, {loc.longitude:.4f})")
                    else:
                        locations.append(ent.text)
                except Exception as e:
                    locations.append(ent.text)
        return ", ".join(list(set(locations))[:3]) if locations else "Not found"

class AdvancedMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        
    def calculate_similarity(self, resume_text, job_desc_text):
        tfidf_matrix = self.vectorizer.fit_transform([resume_text, job_desc_text])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Cached functions for loading resources
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_md"), None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_skills():
    if not os.path.exists("skills.json"):
        return [], "skills.json not found. Please upload it to the app directory."
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            skill_data = json.load(f)
            flat_skills = []
            for category, subcategories in skill_data.items():
                if isinstance(subcategories, dict):
                    for subcat, skills in subcategories.items():
                        flat_skills.extend(skills)
                else:
                    flat_skills.extend(subcategories)
            variations = {
                "NLP": ["Natural Language Processing"],
                "CI/CD": ["Continuous Integration/Continuous Deployment"],
                "SEO": ["Search Engine Optimization"],
                "AI": ["Artificial Intelligence"],
                "ML": ["Machine Learning"]
            }
            expanded_skills = []
            for skill in flat_skills:
                expanded_skills.append(skill)
                if skill in variations:
                    expanded_skills.extend(variations[skill])
            return list(set(expanded_skills)), None
    except Exception as e:
        return [], str(e)

# Define other functions with parameters for nlp and phrase_matcher
def extract_text(file):
    try:
        if file.type == "application/pdf":
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                return "\n".join([page.get_text("text") for page in doc])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_candidate_name(text, nlp):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name_pattern = r"^[A-Za-zÀ-ÿ\-\.']+(?: [A-Za-zÀ-ÿ\-\.']+){1,3}$"
    for line in lines[:5]:
        if re.match(name_pattern, line):
            return line.title()
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return persons[0] if persons else lines[0] if lines else "Unknown"

def extract_section(text, header):
    pattern = re.compile(
        rf'(?i){re.escape(header)}[\s:]*(.*?)(?=\n\s*[A-Z]{{3,}}|\Z)',
        re.DOTALL
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def extract_candidate_summary(resume_text, nlp, parser):
    name = extract_candidate_name(resume_text, nlp)
    tech_skills = extract_section(resume_text, "TECHNICAL SKILLS")
    education = " ".join(parser.extract_education(resume_text))
    work_experience = extract_section(resume_text, "WORK EXPERIENCE")
    certifications = extract_section(resume_text, "CERTIFICATIONS")
    coding_profiles = ""
    profile_keywords = ["leetcode", "codechef", "codeforces", "github"]
    lower_text = resume_text.lower()
    for keyword in profile_keywords:
        if keyword in lower_text:
            coding_profiles += f" {keyword.capitalize()} "
    summary = f"{name}. {tech_skills}. {education}. {work_experience}. {certifications}. {coding_profiles}"
    return summary

def extract_skills(text, nlp, phrase_matcher):
    doc = nlp(text.lower())
    matches = phrase_matcher(doc)
    detected_skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text.title()
        detected_skills.add(skill)
    return {"skills_list": list(detected_skills), "categories": []}

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(title, fontsize=20)
    plt.axis("off")
    st.pyplot(plt)

def generate_pdf_report(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Analysis Report", ln=1, align="C")
    for candidate in results:
        pdf.cell(200, 10, txt=f"Candidate: {candidate['name']}", ln=1)
        pdf.cell(200, 10, txt=f"Match Score: {candidate['score']}", ln=1)
        pdf.cell(200, 10, txt=f"Experience: {candidate['experience']}", ln=1)
        pdf.cell(200, 10, txt=f"Education: {candidate['education']}", ln=1)
        pdf.cell(200, 10, txt=f"Skills: {candidate['skills']}", ln=1)
        pdf.cell(200, 10, txt=f"Categories: {candidate['categories']}", ln=1)
        pdf.cell(200, 10, txt=f"Location: {candidate['location']}", ln=1)
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin1")

# Main function
def main():
    # Set page config as the FIRST Streamlit command
    st.set_page_config(page_title="AI Resume Analyst", layout="wide")
    
    # Load resources AFTER setting page config
    nlp, nlp_error = load_nlp_model()
    if nlp_error:
        st.error(f"Failed to load spaCy model: {nlp_error}")
        st.stop()
    
    skill_list, skills_error = load_skills()
    if skills_error:
        st.error(f"Critical skills error: {skills_error}")
        st.stop()
    
    # Initialize PhraseMatcher
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill.lower()) for skill in skill_list]
    phrase_matcher.add("SKILLS", None, *patterns)
    
    # Geolocation setup
    geolocator = Nominatim(user_agent="resume_parser_app_v2", timeout=20)
    
    # Create parser and matcher
    parser = EnhancedResumeParser(nlp)
    matcher = AdvancedMatcher()
    
    st.title("🚀 AI Resume Analyst 2.0")
    
    st.sidebar.header("⚙️ Analysis Settings")
    st.sidebar.write("Using advanced matching with semantic, TF-IDF, and skills overlap.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_resumes = st.file_uploader("📁 Upload Resumes (PDF/DOCX)", 
                                            type=["pdf", "docx"],
                                            accept_multiple_files=True)
        
    with col2:
        job_description = st.file_uploader("📑 Upload Job Description (for matching)", 
                                           type=["pdf", "docx"])
        min_score = st.slider("🔍 Minimum Match Score", 0.0, 1.0, 0.5)
    
    if uploaded_resumes:
        if st.button("📄 Parse Resumes", key="parse_btn"):
            st.header("📝 Parsed Resume Details")
            parsed_results = []
            for resume in uploaded_resumes:
                resume_text = extract_text(resume)
                candidate_name = extract_candidate_name(resume_text, nlp)
                skills_data = extract_skills(resume_text, nlp, phrase_matcher)
                details = {
                    "Name": candidate_name,
                    "Experience": parser.extract_experience(resume_text),
                    "Education": ", ".join(parser.extract_education(resume_text)),
                    "Skills": ", ".join(skills_data["skills_list"]),
                    "Categories": ", ".join(skills_data["categories"]),
                    "Location": parser.extract_location(resume_text)
                }
                parsed_results.append(details)
                st.subheader(candidate_name)
                st.write(details)
                st.write("**Candidate Summary:**")
                st.write(extract_candidate_summary(resume_text, nlp, parser))
            st.success("Resume parsing complete.")
    
    if uploaded_resumes and job_description:
        if st.button("🔍 Match Resumes", key="match_btn"):
            job_desc_text = extract_text(job_description)
            results = []
            for resume in uploaded_resumes:
                resume_text = extract_text(resume)
                score = matcher.calculate_similarity(resume_text, job_desc_text)
                if score >= min_score:
                    skills_data = extract_skills(resume_text, nlp, phrase_matcher)
                    results.append({
                        "name": extract_candidate_name(resume_text, nlp),
                        "score": round(score, 2),
                        "experience": parser.extract_experience(resume_text),
                        "education": ", ".join(parser.extract_education(resume_text)),
                        "skills": ", ".join(skills_data["skills_list"]),
                        "categories": ", ".join(skills_data["categories"]),
                        "location": parser.extract_location(resume_text),
                        "text": resume_text
                    })
            
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            if results:
                st.header("📊 Analysis Dashboard")
                st.subheader("📈 Global Score Distribution")
                df = pd.DataFrame(results)
                st.bar_chart(df.set_index("name")["score"])
                
                st.subheader("🏆 Candidate Analyses")
                for candidate in results:
                    with st.expander(f"{candidate['name']} - Match Score: {candidate['score']:.0%}"):
                        st.write(f"**Experience:** {candidate['experience']}")
                        st.write(f"**Education:** {candidate['education']}")
                        st.write(f"**Skills:** {candidate['skills']}")
                        st.write(f"**Categories:** {candidate['categories']}")
                        st.write(f"**Location:** {candidate['location']}")
                        
                        candidate_skills = set(extract_skills(candidate['text'], nlp, phrase_matcher)["skills_list"])
                        job_skills = set(extract_skills(job_desc_text, nlp, phrase_matcher)["skills_list"])
                        missing_skills = job_skills - candidate_skills
                        st.write("**Skill Gap Analysis:**")
                        if missing_skills:
                            st.write(f"Missing Skills: {', '.join(missing_skills)}")
                        else:
                            st.write("No missing skills!")
                        
                        st.write("**Keyword Comparison:**")
                        col_kw1, col_kw2 = st.columns(2)
                        with col_kw1:
                            plot_wordcloud(candidate['text'], f"{candidate['name']} Resume Keywords")
                        with col_kw2:
                            plot_wordcloud(job_desc_text, "Job Description Keywords")
                
                st.subheader("🔍 Global Skill Gap Analysis")
                global_job_skills = set(extract_skills(job_desc_text, nlp, phrase_matcher)["skills_list"])
                all_resume_skills = set().union(*[set(r['skills'].split(", ")) for r in results])
                global_missing = global_job_skills - all_resume_skills
                if global_missing:
                    st.warning(f"Missing skills across all candidates: {', '.join(global_missing)}")
                else:
                    st.success("All required skills covered in candidate pool!")
                
                st.subheader("📚 Global Keyword Comparison")
                col_global1, col_global2 = st.columns(2)
                with col_global1:
                    plot_wordcloud(job_desc_text, "Job Description Keywords")
                with col_global2:
                    plot_wordcloud(" ".join([r["text"] for r in results]), "Combined Resume Keywords")
                
                st.subheader("💾 Export Results")
                if st.button("📥 Download Analysis Report"):
                    pdf = generate_pdf_report(results)
                    st.download_button("Download PDF", pdf, file_name="resume_analysis.pdf")
            else:
                st.warning("No candidates meet the minimum score criteria.")

if __name__ == "__main__":
    main()