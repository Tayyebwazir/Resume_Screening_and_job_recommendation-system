import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .confidence-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .skill-tag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .example-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Job recommendations database
JOB_RECOMMENDATIONS = {
    "Data Science": {
        "jobs": [
            "Data Scientist", "Machine Learning Engineer", "AI Researcher", 
            "Business Intelligence Analyst", "Data Analyst", "Research Scientist"
        ],
        "skills": [
            "Python", "R", "SQL", "Machine Learning", "Deep Learning", 
            "Statistics", "Data Visualization", "TensorFlow", "PyTorch", "Pandas"
        ],
        "companies": [
            "Google", "Microsoft", "Amazon", "Meta", "Netflix", 
            "Uber", "Airbnb", "Spotify", "Tesla", "OpenAI"
        ],
        "salary_range": "$80,000 - $180,000",
        "growth_rate": "22% (Much faster than average)"
    },
    "Engineering": {
        "jobs": [
            "Software Engineer", "DevOps Engineer", "Systems Engineer", 
            "Full Stack Developer", "Backend Engineer", "Cloud Engineer"
        ],
        "skills": [
            "Python", "Java", "JavaScript", "C++", "Docker", "Kubernetes", 
            "AWS", "Git", "Linux", "Agile", "System Design"
        ],
        "companies": [
            "Google", "Apple", "Microsoft", "Amazon", "Facebook", 
            "Netflix", "Uber", "Spotify", "GitHub", "Atlassian"
        ],
        "salary_range": "$70,000 - $160,000",
        "growth_rate": "17% (Much faster than average)"
    },
    "Web Designing": {
        "jobs": [
            "UI/UX Designer", "Frontend Developer", "Web Designer", 
            "Product Designer", "Visual Designer", "Interaction Designer"
        ],
        "skills": [
            "HTML", "CSS", "JavaScript", "React", "Vue.js", "Figma", 
            "Adobe Creative Suite", "Responsive Design", "User Experience"
        ],
        "companies": [
            "Adobe", "Figma", "Sketch", "InVision", "Dribbble", 
            "Behance", "Canva", "WordPress", "Squarespace", "Webflow"
        ],
        "salary_range": "$50,000 - $120,000",
        "growth_rate": "8% (Faster than average)"
    },
    "HR": {
        "jobs": [
            "HR Manager", "Talent Acquisition Specialist", "HR Business Partner", 
            "Compensation Analyst", "Training Specialist", "Employee Relations Manager"
        ],
        "skills": [
            "Communication", "Leadership", "Conflict Resolution", "HRIS", 
            "Recruiting", "Performance Management", "Employment Law", "Analytics"
        ],
        "companies": [
            "LinkedIn", "Indeed", "Glassdoor", "Workday", "ADP", 
            "Paychex", "BambooHR", "Zenefits", "Gusto", "Rippling"
        ],
        "salary_range": "$45,000 - $110,000",
        "growth_rate": "7% (Faster than average)"
    },
    "Advocate": {
        "jobs": [
            "Corporate Lawyer", "Legal Consultant", "Compliance Officer", 
            "Contract Specialist", "Legal Analyst", "Paralegal"
        ],
        "skills": [
            "Legal Research", "Contract Law", "Litigation", "Compliance", 
            "Legal Writing", "Negotiation", "Case Management", "Ethics"
        ],
        "companies": [
            "BigLaw Firms", "Corporate Legal Departments", "Government Agencies", 
            "Legal Tech Companies", "Consulting Firms", "Non-profits"
        ],
        "salary_range": "$60,000 - $200,000",
        "growth_rate": "6% (As fast as average)"
    },
    "Business Analyst": {
        "jobs": [
            "Business Analyst", "Product Manager", "Strategy Consultant", 
            "Process Analyst", "Requirements Analyst", "Data Analyst"
        ],
        "skills": [
            "Data Analysis", "SQL", "Excel", "Business Intelligence", 
            "Process Mapping", "Requirements Gathering", "Stakeholder Management"
        ],
        "companies": [
            "McKinsey", "BCG", "Bain", "Deloitte", "PwC", 
            "Accenture", "IBM", "Salesforce", "Oracle", "SAP"
        ],
        "salary_range": "$60,000 - $140,000",
        "growth_rate": "11% (Much faster than average)"
    },
    "Testing": {
        "jobs": [
            "QA Engineer", "Test Automation Engineer", "Performance Tester", 
            "Security Tester", "Mobile Testing Specialist", "DevOps Test Engineer"
        ],
        "skills": [
            "Test Automation", "Selenium", "JIRA", "API Testing", 
            "Performance Testing", "Security Testing", "CI/CD", "Agile"
        ],
        "companies": [
            "Microsoft", "Google", "Amazon", "Adobe", "Oracle", 
            "Salesforce", "Atlassian", "ServiceNow", "Workday", "Zoom"
        ],
        "salary_range": "$55,000 - $120,000",
        "growth_rate": "9% (Faster than average)"
    },
    "DevOps Engineer": {
        "jobs": [
            "DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", 
            "Infrastructure Engineer", "Platform Engineer", "Automation Engineer"
        ],
        "skills": [
            "Docker", "Kubernetes", "AWS", "Azure", "Jenkins", "Terraform", 
            "Ansible", "Monitoring", "CI/CD", "Linux", "Python", "Bash"
        ],
        "companies": [
            "Amazon", "Google", "Microsoft", "Netflix", "Uber", 
            "Airbnb", "Spotify", "GitHub", "HashiCorp", "Red Hat"
        ],
        "salary_range": "$80,000 - $170,000",
        "growth_rate": "20% (Much faster than average)"
    }
}

# Function to clean resume text
def cleanResume(txt):
    cleantxt = re.sub('http\S+\s*', ' ', txt)
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('#\S+', '', cleantxt)
    cleantxt = re.sub('@\S+', '  ', cleantxt)
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]',r' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    return cleantxt

# Load models
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            rf_classifier = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return rf_classifier, tfidf_vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Using demo mode with sample predictions.")
        return None, None

# Prediction function
def predict_category(resume_text, rf_classifier, tfidf_vectorizer):
    if rf_classifier is None or tfidf_vectorizer is None:
        # Demo mode - return sample prediction
        categories = list(JOB_RECOMMENDATIONS.keys())
        predicted_category = np.random.choice(categories)
        confidence = np.random.uniform(75, 95)
        category_probs = {cat: np.random.uniform(5, 95) if cat == predicted_category else np.random.uniform(1, 30) for cat in categories}
        return predicted_category, confidence, category_probs
    
    cleaned_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer.transform([cleaned_text])
    predicted_category = rf_classifier.predict(resume_tfidf)[0]
    prediction_proba = rf_classifier.predict_proba(resume_tfidf)[0]
    
    # Get confidence score
    confidence = max(prediction_proba) * 100
    
    # Get all categories with their probabilities
    categories = rf_classifier.classes_
    category_probs = {cat: prob * 100 for cat, prob in zip(categories, prediction_proba)}
    
    return predicted_category, confidence, category_probs

# Job recommendation function
def get_job_recommendations(predicted_category, confidence_score):
    if predicted_category not in JOB_RECOMMENDATIONS:
        return None
    
    recommendations = JOB_RECOMMENDATIONS[predicted_category]
    
    # Add confidence-based filtering
    if confidence_score > 80:
        status = "🎯 Excellent Match!"
        recommendation_strength = "Highly Recommended"
    elif confidence_score > 60:
        status = "✅ Good Match"
        recommendation_strength = "Recommended"
    else:
        status = "⚠️ Moderate Match"
        recommendation_strength = "Consider with Additional Skills"
    
    return {
        "category": predicted_category,
        "status": status,
        "strength": recommendation_strength,
        "confidence": confidence_score,
        **recommendations
    }

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🎯 AI Job Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Resume Analysis & Career Guidance</div>', unsafe_allow_html=True)
    
    # Load models
    rf_classifier, tfidf_vectorizer = load_models()
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🎯 Job Recommendations", 
        "📊 Resume Analysis", 
        "🔍 Career Insights", 
        "📚 Example Resumes",
        "ℹ️ About"
    ])
    
    if page == "🎯 Job Recommendations":
        job_recommendation_page(rf_classifier, tfidf_vectorizer)
    elif page == "📊 Resume Analysis":
        resume_analysis_page(rf_classifier, tfidf_vectorizer)
    elif page == "🔍 Career Insights":
        career_insights_page()
    elif page == "📚 Example Resumes":
        examples_page(rf_classifier, tfidf_vectorizer)
    elif page == "ℹ️ About":
        about_page()

def job_recommendation_page(rf_classifier, tfidf_vectorizer):
    st.header("🎯 AI Job Recommendations")
    st.write("Get personalized job recommendations based on your resume analysis")
    
    # Input methods
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_method = st.radio("Choose input method:", ["📝 Type Resume", "📁 Upload File"])
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Categories Available", len(JOB_RECOMMENDATIONS))
        st.metric("Job Roles", sum(len(cat["jobs"]) for cat in JOB_RECOMMENDATIONS.values()))
    
    resume_text = ""
    
    if input_method == "📝 Type Resume":
        resume_text = st.text_area(
            "Enter your resume text:",
            height=300,
            placeholder="Paste your complete resume content here...\n\nInclude:\n- Professional summary\n- Work experience\n- Skills\n- Education\n- Projects"
        )
    else:
        uploaded_file = st.file_uploader("Upload your resume file", type=['txt', 'pdf', 'docx'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                resume_text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("📄 PDF and DOCX parsing not implemented in this demo. Please use text files or copy-paste the content.")
    
    if st.button("🚀 Get Job Recommendations", type="primary", use_container_width=True):
        if resume_text.strip():
            with st.spinner("🔍 Analyzing your resume and finding the best job matches..."):
                # Get prediction
                predicted_category, confidence, category_probs = predict_category(
                    resume_text, rf_classifier, tfidf_vectorizer
                )
                
                if predicted_category:
                    # Get job recommendations
                    recommendations = get_job_recommendations(predicted_category, confidence)
                    
                    if recommendations:
                        # Display main recommendation
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h1>Best Career Match: {recommendations['category']}</h1>
                            <h3>{recommendations['status']}</h3>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">
                                Recommendation Strength: {recommendations['strength']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence
                        st.markdown(f"""
                        <div class="confidence-box">
                            <h3>Match Confidence: {confidence:.1f}%</h3>
                            <p>Based on resume content analysis and job market data</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Job recommendations section
                        st.markdown("## 💼 Recommended Job Roles")
                        
                        job_cols = st.columns(3)
                        for i, job in enumerate(recommendations["jobs"][:6]):
                            with job_cols[i % 3]:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4 style="color: #333; margin: 0;">🎯 {job}</h4>
                                    <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0;">
                                        High demand • Good match
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Skills section
                        st.markdown("## 🛠️ Key Skills for This Field")
                        skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in recommendations["skills"][:10]])
                        st.markdown(f'<div style="margin: 1rem 0;">{skills_html}</div>', unsafe_allow_html=True)
                        
                        # Market insights
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>💰 Salary Range</h4>
                                <p style="font-size: 1.2rem; color: #667eea; font-weight: bold;">
                                    {recommendations['salary_range']}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📈 Job Growth</h4>
                                <p style="font-size: 1.2rem; color: #667eea; font-weight: bold;">
                                    {recommendations['growth_rate']}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🏢 Top Companies</h4>
                                <p style="font-size: 1rem; color: #667eea; font-weight: bold;">
                                    {', '.join(recommendations['companies'][:3])}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Top companies
                        st.markdown("## 🏢 Top Hiring Companies")
                        companies_cols = st.columns(5)
                        for i, company in enumerate(recommendations["companies"][:10]):
                            with companies_cols[i % 5]:
                                st.markdown(f"""
                                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 0.5rem 0;">
                                    <p style="font-weight: bold; color: #333; margin: 0;">{company}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Alternative recommendations
                        st.markdown("## 🔄 Alternative Career Paths")
                        sorted_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[1:4]
                        
                        alt_cols = st.columns(3)
                        for i, (alt_category, alt_prob) in enumerate(sorted_probs):
                            with alt_cols[i]:
                                if alt_category in JOB_RECOMMENDATIONS:
                                    st.markdown(f"""
                                    <div class="recommendation-card">
                                        <h4 style="color: #333; margin: 0;">💡 {alt_category}</h4>
                                        <p style="color: #666; margin: 0.5rem 0;">Match: {alt_prob:.1f}%</p>
                                        <p style="font-size: 0.9rem; color: #888; margin: 0;">
                                            {', '.join(JOB_RECOMMENDATIONS[alt_category]['jobs'][:2])}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Action items
                        st.markdown("## 📋 Next Steps")
                        st.markdown(f"""
                        <div class="example-box">
                            <h4>🎯 To strengthen your profile for {recommendations['category']} roles:</h4>
                            <ul>
                                <li>✅ <strong>Highlight relevant skills:</strong> Emphasize {', '.join(recommendations['skills'][:3])} in your resume</li>
                                <li>🔧 <strong>Develop missing skills:</strong> Consider learning {', '.join(recommendations['skills'][3:6])}</li>
                                <li>🏢 <strong>Target companies:</strong> Apply to {', '.join(recommendations['companies'][:3])}</li>
                                <li>📈 <strong>Stay updated:</strong> Follow industry trends and continue learning</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
        else:
            st.warning("⚠️ Please enter your resume text to get personalized job recommendations.")

def resume_analysis_page(rf_classifier, tfidf_vectorizer):
    st.header("📊 Resume Analysis")
    st.write("Detailed analysis of your resume with category probabilities")
    
    # Input section
    resume_text = st.text_area(
        "Enter resume text for analysis:",
        height=250,
        placeholder="Paste your resume content here for detailed analysis..."
    )
    
    if st.button("🔍 Analyze Resume", type="primary"):
        if resume_text.strip():
            with st.spinner("Analyzing resume..."):
                predicted_category, confidence, category_probs = predict_category(
                    resume_text, rf_classifier, tfidf_vectorizer
                )
                
                if predicted_category:
                    # Main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>Primary Category: {predicted_category}</h2>
                        <h3>Confidence: {confidence:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    st.subheader("📊 Category Probabilities")
                    
                    # Create visualization
                    categories = list(category_probs.keys())
                    probabilities = list(category_probs.values())
                    
                    fig = px.bar(
                        x=categories, 
                        y=probabilities,
                        title="Prediction Probabilities by Category",
                        labels={'x': 'Job Categories', 'y': 'Probability (%)'},
                        color=probabilities,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        showlegend=False,
                        xaxis_tickangle=-45,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top matches
                    st.subheader("🏆 Top Category Matches")
                    sorted_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (cat, prob) in enumerate(sorted_probs[:5]):
                        if i == 0:
                            st.success(f"🥇 **{cat}**: {prob:.1f}% - Primary Match")
                        elif i == 1:
                            st.info(f"🥈 **{cat}**: {prob:.1f}% - Secondary Match")
                        elif i == 2:
                            st.warning(f"🥉 **{cat}**: {prob:.1f}% - Third Match")
                        else:
                            st.write(f"#{i+1} **{cat}**: {prob:.1f}%")
        else:
            st.warning("Please enter resume text for analysis.")

def career_insights_page():
    st.header("🔍 Career Market Insights")
    st.write("Explore different career paths and market trends")
    
    # Select category for insights
    selected_category = st.selectbox("Select a career category:", list(JOB_RECOMMENDATIONS.keys()))
    
    if selected_category:
        insights = JOB_RECOMMENDATIONS[selected_category]
        
        # Category overview
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Career Insights: {selected_category}</h2>
            <p style="font-size: 1.1rem;">Comprehensive overview of opportunities and requirements</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💼 Available Roles</h4>
                <p style="font-size: 2rem; color: #667eea; font-weight: bold;">
                    {len(insights['jobs'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🛠️ Key Skills</h4>
                <p style="font-size: 2rem; color: #667eea; font-weight: bold;">
                    {len(insights['skills'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏢 Top Companies</h4>
                <p style="font-size: 2rem; color: #667eea; font-weight: bold;">
                    {len(insights['companies'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed sections
        st.markdown("## 💼 Job Roles")
        job_cols = st.columns(2)
        for i, job in enumerate(insights["jobs"]):
            with job_cols[i % 2]:
                st.markdown(f"• **{job}**")
        
        st.markdown("## 🛠️ Essential Skills")
        skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in insights["skills"]])
        st.markdown(f'<div style="margin: 1rem 0;">{skills_html}</div>', unsafe_allow_html=True)
        
        st.markdown("## 📈 Market Information")
        market_col1, market_col2 = st.columns(2)
        
        with market_col1:
            st.markdown(f"""
            <div class="example-box">
                <h4>💰 Salary Range</h4>
                <p style="font-size: 1.3rem; color: #667eea; font-weight: bold;">
                    {insights['salary_range']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with market_col2:
            st.markdown(f"""
            <div class="example-box">
                <h4>📊 Job Growth Rate</h4>
                <p style="font-size: 1.3rem; color: #667eea; font-weight: bold;">
                    {insights['growth_rate']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## 🏢 Top Hiring Companies")
        companies_cols = st.columns(3)
        for i, company in enumerate(insights["companies"]):
            with companies_cols[i % 3]:
                st.markdown(f"• **{company}**")

def examples_page(rf_classifier, tfidf_vectorizer):
    st.header("📚 Example Resumes")
    st.write("Try these sample resumes to see how the system works")
    
    examples = {
        "Data Scientist": """
        John Smith
        Email: john.smith@email.com | Phone: (555) 123-4567
        
        PROFESSIONAL SUMMARY
        Results-driven Data Scientist with 5+ years of experience in machine learning, statistical analysis, and data visualization. Proven track record of developing predictive models and extracting actionable insights from complex datasets.
        
        TECHNICAL SKILLS
        • Programming: Python, R, SQL, Scala
        • Machine Learning: Scikit-learn, TensorFlow, PyTorch, Keras
        • Data Visualization: Matplotlib, Seaborn, Plotly, Tableau
        • Big Data: Apache Spark, Hadoop, Hive
        • Cloud Platforms: AWS, Google Cloud Platform, Azure
        • Statistics: Regression Analysis, Hypothesis Testing, A/B Testing
        
        PROFESSIONAL EXPERIENCE
        Senior Data Scientist | Tech Corp | 2020-Present
        • Developed machine learning models that improved customer retention by 25%
        • Built real-time recommendation systems serving 1M+ users daily
        • Led cross-functional teams to deliver data-driven solutions
        
        Data Analyst | Analytics Inc | 2018-2020
        • Performed statistical analysis on large datasets to identify business opportunities
        • Created automated reporting dashboards using Python and Tableau
        • Collaborated with stakeholders to translate business requirements into analytical solutions
        
        EDUCATION
        M.S. in Data Science | Stanford University | 2018
        B.S. in Computer Science | UC Berkeley | 2016
        
        PROJECTS
        • Customer Churn Prediction Model: Achieved 92% accuracy using ensemble methods
        • Sales Forecasting System: Reduced forecast error by 30% using time series analysis