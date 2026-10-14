import re
import io
import pandas as pd
import streamlit as st

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Screening AI")
st.write(
    "Upload resumes, enter a job description, "
    "and rank candidates based on their relevance."
)

SKILLS = [
    "python", "java", "javascript", "html", "css",
    "sql", "mysql", "mongodb", "c++", "c#",
    "react", "angular", "node.js", "flask", "django",
    "machine learning", "deep learning", "artificial intelligence",
    "data analysis", "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "aws", "azure", "google cloud", "docker", "kubernetes",
    "linux", "windows server", "networking",
    "tcp/ip", "dns", "dhcp", "git", "github",
    "cybersecurity", "excel", "power bi", "tableau",
    "communication", "teamwork", "problem solving",
    "project management", "testing", "rest api"
]

def extract_resume_text(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            pdf = PdfReader(io.BytesIO(uploaded_file.getvalue()))

            if pdf.is_encrypted:
                try:
                    pdf.decrypt("")
                except Exception:
                    return ""

            pages = []

            for page in pdf.pages:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)

        elif filename.endswith(".txt"):
            return uploaded_file.getvalue().decode(
                "utf-8", errors="ignore"
            )

    except Exception:
        return ""

    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def extract_skills(text):
    cleaned = clean_text(text)
    found_skills = []

    for skill in SKILLS:
        pattern = (
            r"(?<![a-z0-9+#.])"
            + re.escape(skill)
            + r"(?![a-z0-9+#.])"
        )

        if re.search(pattern, cleaned):
            found_skills.append(skill)

    return found_skills


def extract_education(text):
    lines = text.splitlines()

    keywords = [
        "education", "bachelor", "master", "b.sc",
        "b.sc.", "b.tech", "m.sc", "m.sc.",
        "m.tech", "bca", "mca", "degree",
        "university", "college", "diploma",
        "12th", "10th", "computer science"
    ]

    results = []

    for line in lines:
        if any(word in line.lower() for word in keywords):
            line = line.strip()

            if line and line not in results:
                results.append(line)

    return results[:10]



def extract_experience(text):
    lines = text.splitlines()

    keywords = [
        "experience", "internship", "intern",
        "worked at", "employment", "professional experience",
        "years of experience"
    ]

    results = []

    for line in lines:
        if any(word in line.lower() for word in keywords):
            line = line.strip()

            if line and line not in results:
                results.append(line)

    return results[:10]



def calculate_similarity(job_description, resume_texts):
    documents = [job_description] + resume_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [0.0] * len(resume_texts)

    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(
        resume_vectors,
        job_vector
    ).flatten()

    return (scores * 100).round(2).tolist()



uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


job_description = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder=(
        "Example: We need a Python developer with SQL, "
        "Flask, Git, problem-solving, and communication skills."
    )
)



if st.button("Screen and Rank Resumes", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one resume.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        candidate_data = []
        resume_texts = []

        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            text = extract_resume_text(uploaded_file)

            if not text.strip():
                st.warning(
                    f"Could not extract text from "
                    f"{uploaded_file.name}. "
                    "It may be scanned or empty."
                )
                continue

            resume_texts.append(text)

            candidate_data.append({
                "Candidate": uploaded_file.name,
                "Resume Text": text,
                "Skills": extract_skills(text),
                "Education": extract_education(text),
                "Experience": extract_experience(text)
            })

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        if not candidate_data:
            st.error(
                "No readable resumes were found. "
                "Please upload text-based PDF or TXT files."
            )

        else:
            
            scores = calculate_similarity(
                job_description,
                [candidate["Resume Text"]
                 for candidate in candidate_data]
            )

            
            for candidate, score in zip(candidate_data, scores):
                candidate["Match Score (%)"] = score

            candidate_data.sort(
                key=lambda candidate:
                    candidate["Match Score (%)"],
                reverse=True
            )
            st.success("Resume screening completed!")

            st.subheader("🏆 Candidate Rankings")

            results = []

            for rank, candidate in enumerate(
                candidate_data, start=1
            ):
                results.append({
                    "Rank": rank,
                    "Candidate": candidate["Candidate"],
                    "Match Score (%)":
                        candidate["Match Score (%)"],
                    "Skills": ", ".join(candidate["Skills"])
                })

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            st.subheader("📋 Candidate Summary Reports")

            for rank, candidate in enumerate(
                candidate_data, start=1
            ):
                with st.expander(
                    f"Rank {rank}: {candidate['Candidate']}"
                ):
                    st.metric(
                        "Resume Match Score",
                        f"{candidate['Match Score (%)']:.2f}%"
                    )

                    st.write("**Extracted Skills**")

                    if candidate["Skills"]:
                        st.write(", ".join(candidate["Skills"]))
                    else:
                        st.write("No listed skills detected.")

                    st.write("**Education-related lines**")

                    if candidate["Education"]:
                        for item in candidate["Education"]:
                            st.write(f"- {item}")
                    else:
                        st.write("No education-related lines detected.")

                    st.write("**Experience-related lines**")

                    if candidate["Experience"]:
                        for item in candidate["Experience"]:
                            st.write(f"- {item}")
                    else:
                        st.write("No experience-related lines detected.")
            st.subheader("⬇️ Download Ranking Report")

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Candidate Ranking CSV",
                data=csv_data,
                file_name="candidate_ranking_report.csv",
                mime="text/csv"
            )
            st.subheader("📊 Candidate Match Score Chart")

            chart_df = results_df.set_index("Candidate")[
                ["Match Score (%)"]
            ]

            st.bar_chart(chart_df)

            st.caption(
                "Scores indicate textual similarity to the job "
                "description, not a guarantee of candidate suitability."
            )

