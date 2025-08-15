from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import asyncio

# 🔧 Patch asyncio for Streamlit thread compatibility
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🔐 Load Gemini API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in .env file.")

# 🌐 Gemini Client Setup
external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🤖 General Agent (No tools needed)
simple_agent = Agent(
    name="Gemini Resume Coach",
    instructions="You are a helpful and professional AI assistant. Respond to user prompts clearly, concisely, and in a career-friendly tone.",
    tools=[],
)

# ⚙️ Utility function to talk to Gemini via Agent
def ask_gemini(prompt: str):
    result = Runner.run_sync(simple_agent, prompt, run_config=config)
    return result.final_output


# ------------------------- UI -------------------------
# 🧠 Streamlit Setup
st.set_page_config(page_title="💼 Resume & Interview AI Coach", layout="centered")
st.title("💼 Resume & Interview AI Coach")
st.markdown("Use AI to create your **Resume**, **Cover Letter**, and practice **Interview Questions** instantly! 🚀")

# 📋 Sidebar: Input Details
st.sidebar.title("🧾 Enter Your Details")
name = st.sidebar.text_input("👤 Full Name")
email = st.sidebar.text_input("📧 Email")
phone = st.sidebar.text_input("📱 Phone Number")
job_title = st.sidebar.text_input("💼 Job Title you're applying for")
skills = st.sidebar.text_area("🛠️ Skills (comma-separated)")
experience = st.sidebar.text_area("📆 Work Experience")
achievements = st.sidebar.text_area("🏆 Achievements or Projects")
summary = st.sidebar.text_area("🧠 Career Summary (optional)", placeholder="AI will auto-generate if left blank.")

# ------------------------- Resume Generator -------------------------
if st.button("📄 Generate Resume"):
    if name and email and phone and job_title and skills:
        with st.spinner("Generating Resume..."):
            prompt = f"""
Generate a professional resume for the following candidate:

Name: {name}
Email: {email}
Phone: {phone}
Job Title: {job_title}
Skills: {skills}
Experience: {experience}
Achievements: {achievements}
Summary: {summary if summary else 'Auto-generate it'}

Return the resume in a clean, ATS-friendly format.
"""
            output = ask_gemini(prompt)
            st.subheader("📄 Your Resume")
            st.markdown(f"```\n{output}\n```")
            st.download_button("⬇️ Download Resume", output, file_name="resume.txt", mime="text/plain")
    else:
        st.warning("⚠️ Please fill all required fields in the sidebar.")

# ------------------------- Cover Letter Generator -------------------------
if st.button("✉️ Generate Cover Letter"):
    if name and job_title and experience:
        with st.spinner("Generating Cover Letter..."):
            prompt = f"""
Write a compelling and professional cover letter for {name}, applying for the position of {job_title}.
Use this experience: {experience}
Keep it concise and impactful.
"""
            output = ask_gemini(prompt)
            st.subheader("✉️ Your Cover Letter")
            st.markdown(f"```\n{output}\n```")
            st.download_button("⬇️ Download Cover Letter", output, file_name="cover_letter.txt", mime="text/plain")
    else:
        st.warning("⚠️ Please enter Name, Job Title, and Experience.")

# ------------------------- Interview Questions -------------------------
st.markdown("---")
st.subheader("🎯 Mock Interview Questions")

if st.button("🎤 Generate Interview Questions"):
    if job_title:
        with st.spinner("Generating Questions..."):
            prompt = f"Generate 5 thoughtful mock interview questions for the job position: {job_title}."
            output = ask_gemini(prompt)
            st.markdown(f"```\n{output}\n```")
    else:
        st.warning("⚠️ Please provide the Job Title.")

# ------------------------- Interview Feedback -------------------------
st.markdown("---")
st.subheader("🧠 Get Feedback on Your Interview Answer")

user_answer = st.text_area("🗣️ Paste your interview answer here:")

if st.button("🔍 Get Feedback"):
    if user_answer:
        with st.spinner("Evaluating your answer..."):
            prompt = f"Provide constructive feedback and improvement tips for the following interview answer:\n\n{user_answer}"
            feedback = ask_gemini(prompt)
            st.success("💡 Feedback:")
            st.markdown(f"```\n{feedback}\n```")
    else:
        st.warning("⚠️ Please paste your answer above.")

# ------------------------- Footer -------------------------
st.markdown("""<hr style="margin-top:3rem;margin-bottom:0.5rem">""", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <b>© 2025 Created by Sreenesh K S ❤️</b><br>
        🔗 <a href="https://www.linkedin.com/in/sreenesh-ks/" target="_blank">LinkedIn</a> |
        💻 <a href="https://github.com/SREENESHKS" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
