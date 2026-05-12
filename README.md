🔮 CareerAI — 4-Stage AI Career Forecasting Pipeline

AI-powered career intelligence for Indian professionals — built with LangGraph, Groq, OpenAI, and NVIDIA.


📌 Overview
CareerAI is a Streamlit web app that predicts career futures, salary outlooks, and strategic roadmaps for the Indian job market using a 4-stage LLM pipeline orchestrated by LangGraph.

🧠 Pipeline Architecture
LangSmith (Tracing)
     │
     ▼
Stage 1 → Career Matcher       (CSV + keyword match)
Stage 2 → Groq LLaMA3-70B     (raw career intelligence)
Stage 3 → OpenAI GPT-4o-mini  (refined career insight)
Stage 4 → NVIDIA Llama-3.3-70B (5-phase career roadmap)

✨ Features

🔍 Career Prediction — ask anything about a career in India
⚔️ Career Comparison — compare two domains head-to-head
📊 Career Explorer — filter, sort, and browse all 36 careers
💰 ₹ Salary in Lakhs — realistic Indian salary ranges
🌗 Dark / Light theme toggle
🔭 LangGraph trace viewer — inspect every pipeline step


🗂 Project Structure
CareerAI/
├── app.py          # Main Streamlit application
├── data.csv        # Career dataset (36 domains)
├── .env            # API keys (not committed)
├── requirements.txt
└── README.md

⚙️ Setup
1. Clone the repo
bashgit clone https://github.com/your-username/careerai.git
cd careerai
2. Install dependencies
bashpip install -r requirements.txt
3. Configure API keys
Create a .env file in the root directory:
envGROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
NVIDIA_API_KEY=your_nvidia_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=CareerAI

All keys are optional — the app gracefully falls back when a key is missing.

4. Run the app
bashstreamlit run app.py

📦 Requirements
streamlit
langchain
langchain-core
langchain-groq
langchain-openai
langchain-nvidia-ai-endpoints
langgraph
langsmith
pandas
python-dotenv

📊 Dataset (data.csv)
36 career domains with the following fields:
ColumnDescriptioncareer_domainCareer titlekeywordsSearch/match keywordsdemand_scoreCurrent market demand (0–100)growth_rateAnnual growth %future_scoreAI-predicted future viability (0–100)avg_salary_inrAverage Indian salary (₹ INR)automation_risk% risk of automationrisk_levelLow / Medium / Highrequired_skillsKey skills (semicolon-separated)summaryOne-line career overview

🔑 API Keys — Where to Get Them
ServiceLinkGroqhttps://console.groq.comOpenAIhttps://platform.openai.comNVIDIAhttps://integrate.api.nvidia.comLangSmithhttps://smith.langchain.com




🙌 Contributing
Pull requests are welcome. For major changes, open an issue first.
