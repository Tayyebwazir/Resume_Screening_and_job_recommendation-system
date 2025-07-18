import streamlit as st
import re
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import docx
from io import BytesIO
try:
    import phonenumbers
    from phonenumbers import geocoder, carrier
except ImportError:
    phonenumbers = None
    geocoder = None
    carrier = None

# Page configuration
st.set_page_config(
    page_title="Screening AI base Resume and Job Recommendation",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .skill-tag {
        background: #2E86AB;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .prediction-box {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .project-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Job recommendations database
JOB_RECOMMENDATIONS = {
    "Data Science": {
        "jobs": ["Data Scientist", "Machine Learning Engineer", "AI Researcher", "Data Analyst", "Business Intelligence Analyst"],
        "skills": ["Python", "R", "SQL", "Machine Learning", "Statistics", "Data Visualization", "TensorFlow", "PyTorch","Deep Learning","Natural Language Processing"],
        "salary_range": "$80,000 - $180,000",
        "description": "Perfect for analytical minds who love working with data and building predictive models."
    },
    "Engineering": {
        "jobs": ["Software Engineer", "Full Stack Developer", "Backend Engineer", "DevOps Engineer", "System Engineer"],
        "skills": ["Python", "Java", "JavaScript", "C++", "Docker", "AWS", "Git", "System Design"],
        "salary_range": "$70,000 - $160,000",
        "description": "Ideal for problem-solvers who enjoy building scalable software solutions."
    },
    "Web Designing": {
        "jobs": ["UI/UX Designer", "Frontend Developer", "Web Designer", "Product Designer", "Visual Designer"],
        "skills": ["HTML", "CSS", "JavaScript", "React", "Figma", "Adobe Creative Suite", "Responsive Design"],
        "salary_range": "$50,000 - $120,000",
        "description": "Great for creative professionals who want to create engaging user experiences."
    },
    "HR": {
        "jobs": ["HR Manager", "Talent Acquisition", "HR Business Partner", "Training Specialist", "Employee Relations"],
        "skills": ["Communication", "Leadership", "HRIS", "Recruiting", "Performance Management", "Employment Law"],
        "salary_range": "$45,000 - $110,000",
        "description": "Perfect for people-oriented professionals who enjoy building organizational culture."
    },
    "Business Analyst": {
        "jobs": ["Business Analyst", "Product Manager", "Strategy Consultant", "Process Analyst", "Data Analyst"],
        "skills": ["Data Analysis", "SQL", "Excel", "Business Intelligence", "Requirements Gathering", "Stakeholder Management"],
        "salary_range": "$60,000 - $140,000",
        "description": "Excellent for strategic thinkers who bridge business and technology."
    },
    "Testing": {
        "jobs": ["QA Engineer", "Test Automation Engineer", "Performance Tester", "Security Tester", "Mobile Testing"],
        "skills": ["Test Automation", "Selenium", "JIRA", "API Testing", "Performance Testing", "Agile"],
        "salary_range": "$55,000 - $120,000",
        "description": "Ideal for detail-oriented professionals who ensure software quality."
    }
}

def cleanResume(txt):
    """Clean and preprocess resume text"""
    cleantxt = re.sub('http\S+\s*', ' ', txt)
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('#\S+', '', cleantxt)
    cleantxt = re.sub('@\S+', '  ', cleantxt)
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', r' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    return cleantxt

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from Word document"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

# Load models (with error handling)
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            rf_classifier = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return rf_classifier, tfidf_vectorizer
    except FileNotFoundError:
        st.warning("⚠️ Model files not found! Using demo mode.")
        return None, None

def extract_name(resume_text):
    """Extract candidate name from resume"""
    lines = resume_text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) < 2 or not any(c.isalpha() for c in line):
            continue
        if any(keyword in line.lower() for keyword in ['email', 'phone', 'tel', '@', 'resume', 'cv', 'summary', 'objective']):
            continue
        
        # Remove common prefixes
        line = re.sub(r'^(mr\.?|ms\.?|mrs\.?|dr\.?)\s+', '', line, flags=re.IGNORECASE)
        
        words = line.split()
        if 2 <= len(words) <= 4 and all(word.replace('.', '').replace('-', '').isalpha() for word in words):
            return line
    return "Name not found"

def extract_contact(resume_text):
    """Extract contact information from resume"""
    contact_info = {}
    
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    contact_info['email'] = emails[0] if emails else "Not found"
    
    # Enhanced phone extraction
    phone_patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # US format
        r'\+?(\d{1,3})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})',    # International
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',  # Simple format
        r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',  # With parentheses
    ]
    
    phone_found = "Not found"
    for pattern in phone_patterns:
        matches = re.findall(pattern, resume_text)
        if matches:
            if isinstance(matches[0], tuple):
                phone_found = ''.join(matches[0])
            else:
                phone_found = matches[0]
            break
    
    # Try to validate and format phone number
    if phonenumbers is not None:
        try:
            if phone_found != "Not found":
                # Clean the phone number
                clean_phone = re.sub(r'[^\d+]', '', phone_found)
                if len(clean_phone) == 10:
                    clean_phone = '+1' + clean_phone
                elif len(clean_phone) == 11 and clean_phone[0] == '1':
                    clean_phone = '+' + clean_phone
                parsed = phonenumbers.parse(clean_phone, None)
                if phonenumbers.is_valid_number(parsed):
                    contact_info['phone'] = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                else:
                    contact_info['phone'] = phone_found
            else:
                contact_info['phone'] = "Not found"
        except:
            contact_info['phone'] = phone_found if phone_found != "Not found" else "Not found"
    else:
        contact_info['phone'] = phone_found if phone_found != "Not found" else "Not found"
    return contact_info

def extract_skills(resume_text):
    """Extract skills from resume"""
    skill_keywords = [
        'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 'vue',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'git', 'github',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
        'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd',
        'agile', 'scrum', 'jira', 'confluence', 'slack', 'communication',
        'leadership', 'project management', 'problem solving', 'teamwork',
        'flask', 'django', 'fastapi', 'node.js', 'express', 'spring',
        'tableau', 'power bi', 'excel', 'statistics', 'analytics',
        'c++', 'c#', '.net', 'php', 'ruby', 'go', 'rust', 'swift',
        'figma', 'adobe', 'photoshop', 'illustrator', 'sketch',
        'selenium', 'automation', 'testing', 'qa', 'performance testing'
    ]
    
    resume_lower = resume_text.lower()
    found_skills = []
    
    for skill in skill_keywords:
        if skill in resume_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))  # Remove duplicates

def extract_projects(resume_text):
    """Extract projects with names from resume"""
    projects = []
    
    # Enhanced project extraction patterns
    project_patterns = [
        r'projects?\s*:?\s*\n(.*?)(?=\n\s*(?:experience|education|skills|certifications|awards|\Z))',
        r'key projects?\s*:?\s*\n(.*?)(?=\n\s*(?:experience|education|skills|certifications|awards|\Z))',
        r'notable projects?\s*:?\s*\n(.*?)(?=\n\s*(?:experience|education|skills|certifications|awards|\Z))',
        r'project experience\s*:?\s*\n(.*?)(?=\n\s*(?:experience|education|skills|certifications|awards|\Z))',
    ]
    
    for pattern in project_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if matches:
            project_text = matches[0].strip()
            
            # Split by common project separators
            project_lines = re.split(r'[•\-\*]|\n(?=\d+\.|\w+:)', project_text)
            
            for line in project_lines:
                line = line.strip()
                if len(line) > 15:  # Filter out very short lines
                    # Try to extract project name and description
                    project_match = re.match(r'^(.+?):\s*(.+)$', line, re.DOTALL)
                    if project_match:
                        project_name = project_match.group(1).strip()
                        project_desc = project_match.group(2).strip()
                        projects.append({
                            'name': project_name,
                            'description': project_desc[:200] + '...' if len(project_desc) > 200 else project_desc
                        })
                    else:
                        # Extract first sentence as potential project name
                        sentences = re.split(r'[.!?]', line)
                        if sentences:
                            project_name = sentences[0].strip()
                            project_desc = line
                            projects.append({
                                'name': project_name[:50] + '...' if len(project_name) > 50 else project_name,
                                'description': project_desc[:200] + '...' if len(project_desc) > 200 else project_desc
                            })
    
    return projects[:5]  # Return top 5 projects

def predict_job_recommendation(resume_text, rf_classifier, tfidf_vectorizer):
    """Predict job category and confidence"""
    if rf_classifier is None or tfidf_vectorizer is None:
        # Demo mode - analyze text to make educated guess
        resume_lower = resume_text.lower()
        scores = {}
        
        # Keyword-based scoring for demo
        for category, info in JOB_RECOMMENDATIONS.items():
            score = 0
            for skill in info['skills']:
                if skill.lower() in resume_lower:
                    score += 1
            scores[category] = score
        
        if scores:
            predicted_category = max(scores, key=scores.get)
            max_score = scores[predicted_category]
            confidence = min(95, max(60, (max_score / len(JOB_RECOMMENDATIONS[predicted_category]['skills'])) * 100))
        else:
            categories = list(JOB_RECOMMENDATIONS.keys())
            predicted_category = np.random.choice(categories)
            confidence = np.random.uniform(65, 85)
            
        return predicted_category, confidence
    
    cleaned_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer.transform([cleaned_text])
    predicted_category = rf_classifier.predict(resume_tfidf)[0]
    prediction_proba = rf_classifier.predict_proba(resume_tfidf)[0]
    confidence = max(prediction_proba) * 100
    
    return predicted_category, confidence

def get_job_recommendations(predicted_category, confidence_score):
    """Get detailed job recommendations"""
    if predicted_category not in JOB_RECOMMENDATIONS:
        return None
    
    recommendations = JOB_RECOMMENDATIONS[predicted_category]
    
    # Determine recommendation strength
    if confidence_score >= 80:
        status = "🎯 Excellent Match!"
        strength = "Highly Recommended"
        message = "This candidate is an excellent fit for this field!"
    elif confidence_score >= 65:
        status = "✅ Good Match"
        strength = "Recommended"
        message = "This candidate shows good potential for this field."
    elif confidence_score >= 50:
        status = "⚠️ Moderate Match"
        strength = "Consider with skill development"
        message = "With some additional training, this candidate could succeed in this field."
    else:
        status = "❌ Low Match"
        strength = "Not recommended"
        message = "This candidate might want to consider other fields or significant skill development."
    
    return {
        "category": predicted_category,
        "status": status,
        "strength": strength,
        "message": message,
        "confidence": confidence_score,
        "jobs": recommendations["jobs"],
        "skills": recommendations["skills"],
        "salary_range": recommendations["salary_range"],
        "description": recommendations["description"]
    }

def main():
    st.markdown('<div class="main-header">🎯 Enhanced Resume Screening AI System</div>', unsafe_allow_html=True)
    st.markdown("AI-powered resume screening with intelligent job recommendations and comprehensive analysis")
    
    # Load models
    rf_classifier, tfidf_vectorizer = load_models()
    
    # Input section
    st.header("📝 Upload Resume for Analysis")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload File (PDF/Word/TXT)", "Type Resume"])
    
    resume_text = ""
    
    if input_method == "Upload File (PDF/Word/TXT)":
        uploaded_file = st.file_uploader("Upload resume file", type=['pdf', 'docx', 'txt'])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                resume_text = extract_text_from_docx(uploaded_file)
            elif file_type == 'txt':
                resume_text = str(uploaded_file.read(), "utf-8")
            
            if resume_text:
                st.success(f"✅ Successfully extracted text from {uploaded_file.name}")
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted text preview:", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200)
    else:
        resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Paste your complete resume content here..."
        )
    
    # Analysis button
    if st.button("🎯 Analyze Resume & Get Job Recommendations", type="primary", use_container_width=True):
        if resume_text.strip():
            with st.spinner("🔍 Analyzing resume and generating recommendations..."):
                # Extract features
                name = extract_name(resume_text)
                contact = extract_contact(resume_text)
                skills = extract_skills(resume_text)
                projects = extract_projects(resume_text)
                
                # Get job recommendations
                predicted_category, confidence = predict_job_recommendation(resume_text, rf_classifier, tfidf_vectorizer)
                recommendations = get_job_recommendations(predicted_category, confidence)
                
                # Display results
                st.header("🎯 Analysis Results")
                
                # Main recommendation with confidence
                if recommendations:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Recommended Job Category</h2>
                        <h1>{recommendations['category']}</h1>
                        <h3>{recommendations['status']}</h3>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            <strong>Confidence Score: {confidence:.1f}%</strong>
                        </p>
                        <p style="font-size: 1rem;">
                            {recommendations['message']}
                        </p>
                        <p style="font-size: 0.9rem; margin-top: 1rem;">
                            {recommendations['description']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Job roles section
                    st.markdown("### 💼 Recommended Job Roles")
                    job_cols = st.columns(3)
                    for i, job in enumerate(recommendations["jobs"]):
                        with job_cols[i % 3]:
                            st.markdown(f"""
                            <div class="feature-box">
                                <h5 style="color: #2E86AB; margin: 0;">🎯 {job}</h5>
                                <p style="color: #666; font-size: 0.9rem;">Match: {confidence:.0f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Skills and salary info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🛠️ Required Skills for This Field")
                        if recommendations["skills"]:
                            skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in recommendations["skills"]])
                            st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="feature-box">
                            <h4>💰 Expected Salary Range</h4>
                            <p style="font-size: 1.3rem; font-weight: bold; color: #2E86AB;">
                                {recommendations['salary_range']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Candidate profile section
                st.markdown("---")
                st.markdown("### 👤 Candidate Profile Summary")
                
                profile_col1, profile_col2 = st.columns(2)
                
                with profile_col1:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>📋 Basic Information</h4>
                        <p><strong>Name:</strong> {name}</p>
                        <p><strong>Email:</strong> {contact['email']}</p>
                        <p><strong>Phone:</strong> {contact['phone']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with profile_col2:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>📊 Profile Metrics</h4>
                        <p><strong>Skills Identified:</strong> {len(skills)}</p>
                        <p><strong>Projects Found:</strong> {len(projects)}</p>
                        <p><strong>Job Match Score:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Skills analysis
                if skills:
                    st.markdown("### 🔍 Candidate Skills Analysis")
                    candidate_skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>Identified Skills ({len(skills)} total)</h4>
                        <div>{candidate_skills_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("### 🔍 Skills Analysis")
                    st.markdown("""
                    <div class="feature-box">
                        <h4>⚠️ No Skills Detected</h4>
                        <p>No recognizable skills were found in the resume. Consider adding more technical skills or using standard skill terminology.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Projects section
                if projects:
                    st.markdown("### 🚀 Project Portfolio")
                    for i, project in enumerate(projects, 1):
                        st.markdown(f"""
                        <div class="project-card">
                            <h5 style="color: #28a745; margin-bottom: 0.5rem;">📁 {project['name']}</h5>
                            <p style="margin: 0; color: #666;">{project['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("### 🚀 Project Portfolio")
                    st.markdown("""
                    <div class="feature-box">
                        <h4>⚠️ No Projects Found</h4>
                        <p>No projects were identified in the resume. Consider adding a projects section to showcase your work.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation summary
                st.markdown("### 📋 Final Screening Summary")
                
                # Color-coded confidence indicator
                if confidence >= 80:
                    confidence_color = "#28a745"
                elif confidence >= 65:
                    confidence_color = "#ffc107"
                elif confidence >= 50:
                    confidence_color = "#fd7e14"
                else:
                    confidence_color = "#dc3545"
                
                st.markdown(f"""
                <div class="feature-box">
                    <h4>🎯 Final Recommendation</h4>
                    <p><strong>Candidate:</strong> {name}</p>
                    <p><strong>Best Job Category:</strong> {predicted_category}</p>
                    <p><strong>Confidence Score:</strong> 
                        <span style="color: {confidence_color}; font-weight: bold; font-size: 1.2rem;">
                            {confidence:.1f}%
                        </span>
                    </p>
                    <p><strong>Recommendation:</strong> {recommendations['strength'] if recommendations else 'N/A'}</p>
                    <p><strong>Contact Info:</strong> {'✅ Complete' if contact['email'] != 'Not found' and contact['phone'] != 'Not found' else '⚠️ Incomplete'}</p>
                    <p><strong>Profile Completeness:</strong> {'✅ Good' if len(skills) > 3 and len(projects) > 0 else '⚠️ Needs Improvement'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export results
                st.markdown("### 💾 Export Analysis Report")
                screening_results = {
                    'Candidate_Name': name,
                    'Email': contact['email'],
                    'Phone': contact['phone'],
                    'Recommended_Job_Category': predicted_category,
                    'Confidence_Score': f"{confidence:.1f}%",
                    'Recommendation_Status': recommendations['status'] if recommendations else 'N/A',
                    'Recommendation_Strength': recommendations['strength'] if recommendations else 'N/A',
                    'Skills_Count': len(skills),
                    'Projects_Count': len(projects),
                    'Top_Skills': ', '.join(skills[:10]),
                    'Salary_Range': recommendations['salary_range'] if recommendations else 'N/A',
                    'Recommendation_Message': recommendations['message'] if recommendations else 'N/A'
                }
                
                df_results = pd.DataFrame([screening_results])
                csv = df_results.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv,
                    file_name=f"resume_analysis_{name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
        else:
            st.warning("⚠️ Please upload a resume file or enter resume text to start the analysis.")

    # Sample resume section
    st.markdown("---")
    st.markdown("### 📝 Try Sample Resume")
    if st.button("Load Sample Resume for Testing"):
        sample_resume = """
John Smith
Senior Data Scientist
Email: john.smith@email.com
Phone: +1 (555) 123-4567
LinkedIn: linkedin.com/in/johnsmith

Professional Summary
Results-driven Data Scientist with 5+ years of experience in machine learning, statistical analysis, and data visualization. Proven track record of developing predictive models and extracting actionable insights from complex datasets to drive business decisions.

Technical Skills
• Programming Languages: Python, R, SQL, Scala, Java
• Machine Learning: Scikit-learn, TensorFlow, PyTorch, Keras, XGBoost
• Data Visualization: Matplotlib, Seaborn, Plotly, Tableau, Power BI
• Big Data Technologies: Apache Spark, Hadoop, Hive, Kafka
• Cloud Platforms: AWS (S3, EC2, Lambda), Google Cloud Platform, Azure
• Statistics: Regression Analysis, Hypothesis Testing, A/B Testing, Time Series Analysis

Professional Experience
Senior Data Scientist | Tech Corp | Jan 2020 - Present
• Developed machine learning models that improved customer retention by 25%
• Built real-time recommendation systems serving 1M+ users daily
• Led cross-functional teams to deliver data-driven solutions reducing costs by $2M annually
• Implemented automated ML pipelines using Apache Airflow and Docker

Data Analyst | Analytics Inc | Jun 2018 - Dec 2019
• Performed statistical analysis on large datasets to identify business opportunities
• Created automated reporting dashboards using Python and Tableau
• Collaborated with stakeholders to translate business requirements into analytical solutions
• Improved data processing efficiency by 40% through optimization techniques

Education
Master of Science in Data Science | Stanford University | 2018
Bachelor of Science in Computer Science | UC Berkeley | 2016

Certifications
• AWS Certified Solutions Architect
• Google Cloud Professional Data Engineer
• Microsoft Azure Data Scientist Associate

Projects
Customer Churn Prediction System: Developed an ensemble model using Random Forest and XGBoost achieving 92% accuracy in predicting customer churn. Deployed the model in production serving real-time predictions.

Sales Forecasting Dashboard: Built a comprehensive time series forecasting system using ARIMA and Prophet models, reducing forecast error by 30%. Created interactive Tableau dashboards for stakeholders.

Real-time Fraud Detection Pipeline: Implemented a machine learning pipeline processing 100K+ transactions daily using Apache Kafka and Spark Streaming. Reduced fraud losses by 45%.

Sentiment Analysis Tool: Created a natural language processing application using BERT and transformer models to analyze customer feedback across multiple channels.

Movie Recommendation Engine: Developed a collaborative filtering recommendation system using matrix factorization techniques, improving user engagement by 20%.
        """
        st.text_area("Sample Resume Content:", value=sample_resume, height=400)

if __name__ == "__main__":
    main()