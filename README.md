# PhishDec: Multi-Agent AI Cybersecurity Analyst 🛡️

![PhishDec Banner](https://img.shields.io/badge/Security-AI%20Powered-38bdf8?style=for-the-badge)
![React](https://img.shields.io/badge/Frontend-React-20232a?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

**PhishDec** is an advanced, automated cybersecurity platform that leverages a **Multi-Agent Artificial Intelligence** architecture to detect, analyze, and mitigate phishing threats in real-time. By orchestrating multiple specialized AI agents, PhishDec performs highly accurate, corroborated security assessments of suspicious URLs and websites, operating exactly like a human SOC (Security Operations Center) team.

---

## 🌟 Key Features

- **Multi-Agent Architecture**: Six independent specialized AI agents working concurrently:
  - 🌐 **URL Agent**: Analyzes lexical features and domain characteristics.
  - 📄 **HTML Agent**: Inspects DOM structure, forms, and obfuscated JavaScript.
  - 🔒 **SSL/TLS Agent**: Validates certificate chains, issuer reputation, and cryptography.
  - 📡 **DNS Agent**: Investigates A/MX/TXT records, fast-flux networks, and domain age.
  - 👁️ **Visual Agent**: Employs Computer Vision (YOLOv8) to detect brand spoofing and logo hijacking.
  - 🚨 **Threat Intel Agent**: Cross-references IOCs with VirusTotal, PhishTank, and Google Safe Browsing.
- **Decision Fusion Engine**: A master orchestrator that mathematically fuses agent predictions to achieve consensus, resolve conflicts, and output a highly reliable final risk score.
- **Role-Based Access Control (RBAC)**: Distinct dashboards and permissions for **Super Admins**, **Admins**, and **Analysts**.
- **Automated PDF Reporting**: Generates stunning, pixel-perfect PDF investigation reports for incident response and compliance auditing.
- **Glassmorphic Cyber UI**: A premium, dark-mode React frontend with glowing borders, real-time analytics, and smooth transitions.

---

## 📂 Project Structure

```text
PhishDec/
├── agents/                 # Core Multi-Agent AI Models & Logic
│   ├── fusion.py           # Decision Fusion Engine
│   ├── orchestrator.py     # Master Orchestrator (Concurrency & State)
│   ├── url_agent/          # URL Lexical Analysis Agent
│   ├── html_agent/         # HTML DOM Analysis Agent
│   ├── dns_agent/          # DNS Record Analysis Agent
│   ├── ssl_agent/          # Certificate Analysis Agent
│   ├── visual_agent/       # YOLOv8 Computer Vision Agent
│   └── threat_agent/       # Threat Intelligence (OSINT) Agent
├── api/                    # Flask Backend API
│   ├── app.py              # Main Flask Application Entry Point
│   ├── routes/             # API Endpoints (Auth, Analytics, Investigations)
│   └── services/           # Business Logic & PDF Generation
├── database/               # SQLite Database schema and repository controllers
├── frontend/               # React (Vite) Frontend Application
│   ├── src/pages/          # UI Views (Dashboards, Admin Panels, Settings)
│   ├── src/components/     # Reusable UI Components
│   └── package.json        # Frontend dependencies
├── templates/              # Jinja2 HTML templates for PDF Reports
├── .env.example            # Environment variables template
├── requirements.txt        # Python backend dependencies
└── start_phishdec.bat      # One-click startup script for Windows
```

---

## 🚀 Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.9+** (For the Flask backend and AI models)
- **Node.js 18+** (For the React frontend)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PhishDec.git
cd PhishDec
```

### 2. Backend Setup (Python)
It is highly recommended to use a virtual environment.
```bash
# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the provided `.env.example` file to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Open `.env` and configure:
```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=YourSuperSecretKeyHere
DATABASE_URL=sqlite:///data/phishdec.db

# API Keys for the Threat Intel Agent (Optional but recommended)
VIRUSTOTAL_API_KEY=your_virustotal_key
PHISHTANK_API_KEY=your_phishtank_key
GOOGLE_SAFE_BROWSING_KEY=your_google_safe_browsing_key
```

### 4. Frontend Setup (React/Node)
Open a new terminal window:
```bash
cd frontend
npm install
```

---

## 🏃‍♂️ Running the Application

### The Easy Way (Windows)
Simply double-click the `start_phishdec.bat` file in the root directory. It will automatically launch both the Flask backend and Vite frontend, and open your browser to the application.

### The Manual Way
**Terminal 1 (Backend):**
```bash
# Ensure venv is activated
python api/app.py
# Backend runs on http://127.0.0.1:5050
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:5173
```

---

## 🛡️ Usage & User Roles

PhishDec includes built-in role segregation. Upon your first installation, the default credentials usually are (check your database seeding logic):

- **Super Admin** (`superadmin@phishdec.com`): Full control over system settings, global alerts, API keys, and comprehensive user activity logs.
- **Admin** (`admin@phishdec.com`): Can manage analyst accounts and view system-wide investigation metrics.
- **Analyst** (`analyst@phishdec.com`): Can submit URLs for real-time investigation, review Multi-Agent telemetry, and export PDF Threat Reports.

### Submitting an Investigation
1. Log in as an Analyst.
2. Navigate to the **Analyze** tab.
3. Enter a suspicious URL.
4. Watch in real-time as the orchestrator dispatches the 6 specialized AI agents.
5. Review the final Fusion Consensus Verdict, Risk Factors, and download the **Security Investigation Report**.

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with passion for next-generation AI cybersecurity.*

## 📃 Dependencies

The full list of backend dependencies is tracked in [requirements.txt](requirements.txt). This includes core ML libraries such as:
- **PyTorch & Ultralytics** (YOLOv8 Computer Vision)
- **XGBoost & Scikit-Learn** (Decision Models)
- **Flask & SQLAlchemy** (Backend API & Database)
- **xhtml2pdf** (Automated Report Generation)
