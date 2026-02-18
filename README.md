# SARAL

SARAL is a civic-tech platform designed to streamline the reporting and management of civic issues, with a focus on road safety and community engagement. The system enables citizens to report violations (such as helmetless riding), track their submissions, and receive rewards, while authorities can efficiently review, process, and analyze reports.

## Features

- **Citizen Reporting:**
  - Submit reports of civic issues (e.g., helmet violations) with photos and location data.
  - Track the status of submitted reports (Pending, Approved, Rejected, Auto-Rejected).
  - View report history and receive rewards for valid submissions.

- **Authority Dashboard:**
  - Review and process incoming reports (excluding system auto-rejected ones).
  - Access analytics and archives for decision-making.
  - Manage authority settings and user accounts.

- **Automated Processing:**
  - System auto-rejects reports where no violation is detected (e.g., everyone is wearing a helmet).
  - Auto-rejected reports are visible only to the reporting citizen.

- **Rewards System:**
  - Citizens earn points for valid reports.
  - Redeem points for rewards via the rewards dashboard.

- **Authentication:**
  - Secure sign-in and sign-up for both citizens and authorities.

## Project Structure

```
SARAL/
├── backend/
│   ├── database.py         # SQLite DB logic and schema
│   ├── main.py             # FastAPI backend server
│   └── uploads/            # Uploaded report images
├── css/                    # Stylesheets
├── js/                     # Frontend JavaScript
├── models/                 # Data models
├── *.html                  # Frontend pages
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore rules
```

## Tech Stack

- **Backend:** Python, FastAPI, SQLite
- **Frontend:** HTML, CSS, Vanilla JavaScript
- **Other:** Git for version control

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Si-cyber-ai/SARAL.git
   cd SARAL
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the backend server:**
   ```sh
   cd backend
   python main.py
   ```
   The backend will start on the default FastAPI port (usually 8000).

4. **Open the frontend:**
   - Open `index.html` or other HTML files in your browser.
   - Ensure the backend is running for API calls to work.

## Key Files

- `backend/main.py` — FastAPI server and API endpoints
- `backend/database.py` — Database schema and queries
- `js/` — Frontend logic (API calls, status mapping, UI updates)
- `css/` — Stylesheets for all pages
- `requirements.txt` — Python dependencies
- `.gitignore` — Files and folders ignored by git

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Civic-tech community inspiration
- FastAPI and open-source contributors
