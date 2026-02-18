# SARAL — Smart Automated Recognition of Automobile Licenses

> **A civic-tech platform for AI-powered traffic violation reporting and community-driven road safety.**

SARAL enables citizens to report traffic violations (such as helmetless riding) by uploading photos or videos. An AI pipeline automatically detects license plates and helmet usage, classifies the violation, and routes the report to the appropriate authority for review. Valid reports earn citizens karma points redeemable for real-world rewards.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [AI Pipeline](#ai-pipeline)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Setup & Installation](#setup--installation)
- [Default Accounts](#default-accounts)
- [Rewards Catalogue](#rewards-catalogue)
- [Frontend Pages](#frontend-pages)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 👤 Citizen Portal
- **Submit Reports** — Upload images or videos of traffic violations with optional location, description, and violation type.
- **AI-Assisted Analysis** — The system automatically detects license plates via YOLOv8 and reads them via EasyOCR. Helmet presence/absence is also detected.
- **Manual Plate Entry** — Citizens can enter the plate number manually; the system cross-validates it against the OCR result.
- **Report Tracking** — View all submitted reports with live status: `Under Review`, `Approved`, `Rejected`, or `Auto-Rejected`.
- **Karma Points** — Earn **150 points** for every approved report.
- **Rewards Redemption** — Redeem karma points for FASTag recharges, metro passes, fuel vouchers, and gift cards.
- **Profile Management** — Update name, email, city, and password.

### 🏛️ Authority Panel
- **Review Queue** — See all reports currently `Under Review` (auto-rejected reports are hidden from authorities).
- **Approve / Reject** — One-click actions with automatic karma award on approval.
- **Analytics Dashboard** — Overview of total, pending, approved, and rejected reports with action percentage.
- **Archive** — Full searchable history of all processed reports.
- **Authority Settings** — Manage authority account details.

### 🤖 Automated Processing
- Reports with AI confidence **below 30%** are automatically set to `Auto-Rejected`.
- Helmet-violation reports where the AI detects **zero helmetless riders** are automatically `Auto-Rejected`.
- Auto-rejected reports are **only visible to the reporting citizen**, not to authorities.
- Annotated images with bounding boxes are generated and stored alongside original uploads.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Database** | SQLite (via `sqlite3`) |
| **AI — Plate Detection** | YOLOv8 via Roboflow Inference SDK |
| **AI — Helmet Detection** | Roboflow-hosted helmet detection model |
| **AI — OCR** | EasyOCR |
| **Image Processing** | OpenCV, NumPy, Matplotlib |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **State Management** | Custom `SaralStore` (localStorage-backed) |
| **Auth** | Session-based via `localStorage` |

---

## Project Structure

```
SARAL/
├── backend/
│   ├── main.py             # FastAPI server — all API routes
│   ├── database.py         # SQLite schema, queries, and migrations
│   ├── saral.db            # SQLite database file (auto-created)
│   └── uploads/            # Uploaded report images & annotated outputs
│
├── models/
│   └── model.py            # Full AI pipeline (detection → OCR → correction)
│
├── css/
│   ├── style.css           # Landing page styles
│   ├── dashboard.css       # Citizen dashboard styles
│   ├── authority.css       # Authority panel styles
│   └── auth.css            # Sign-in / sign-up styles
│
├── js/
│   ├── api.js              # Centralized API client (SaralAPI)
│   ├── auth.js             # Auth & RBAC (SaralAuth, SaralNav)
│   ├── store.js            # Global state store & toast system (SaralStore, SaralToast)
│   ├── main.js             # Landing page logic
│   ├── dashboard.js        # Citizen dashboard logic
│   ├── my-reports.js       # Report history page logic
│   ├── rewards.js          # Rewards page logic
│   └── settings.js         # Settings page logic
│
├── index.html              # Public landing page
├── signin.html             # Sign-in page
├── signup.html             # Sign-up page
├── dashboard.html          # Citizen dashboard
├── report.html             # Report submission form
├── my-reports.html         # Report history
├── rewards.html            # Rewards redemption
├── settings.html           # User settings
├── authority.html          # Authority review queue
├── authority-analytics.html # Authority analytics
├── authority-archive.html  # Authority report archive
├── authority-settings.html # Authority settings
│
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## AI Pipeline

The AI pipeline in `models/model.py` processes each uploaded image or video through the following stages:

```
Upload
  │
  ▼
Stage 0 ── Helmet Detection (Roboflow YOLOv8)
  │         Detects riders with/without helmets
  │
  ▼
Stage 1 ── License Plate Detection (Roboflow YOLOv8 — "indian-plate/1")
  │         Returns bounding boxes for all detected plates
  │
  ▼
Stage 2 ── Crop & Pad
  │         Each detected plate region is cropped with 15% padding
  │
  ▼
Stage 3a ── Image Preprocessing (8 variants per crop)
  │          CLAHE+Otsu, Adaptive Threshold, Morphological,
  │          Inverted, Sharpened, Bilateral, Strong CLAHE, Gaussian Adaptive
  │
  ▼
Stage 3b ── EasyOCR Text Reading
  │          Reads text from all 8 variants; picks best by scoring function
  │          (OCR confidence + Indian plate format bonus − noise penalty)
  │
  ▼
Stage 3c ── OCR Error Correction
  │          Position-aware character confusion maps (digit↔letter)
  │          Exhaustive district/series/number split scoring
  │          State code validation against all 37 Indian RTO codes
  │
  ▼
Stage 4 ── Video Majority Voting (video inputs only)
  │         N frames sampled evenly; majority vote picks final plate text
  │
  ▼
Result ── plate_text, confidence, helmet_detections, annotated_image
```

### Graceful Fallback
If AI dependencies are unavailable at startup, the backend continues to run. The `/api/analyze` endpoint returns a mock result (`KL 11 AB 1234`, 91% confidence) so the rest of the application remains functional.

---

## API Reference

All endpoints are served by FastAPI at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/signin` | Sign in with email, password, role |
| `POST` | `/api/auth/signup` | Register a new account |
| `GET` | `/api/auth/user/{user_id}` | Get user profile |
| `PUT` | `/api/auth/user/{user_id}` | Update user profile |

### Analysis & Reports

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload media, run AI pipeline, create report |
| `GET` | `/api/reports/user/{user_id}` | Get all reports for a citizen |
| `GET` | `/api/reports/stats/{user_id}` | Get report statistics for a citizen |
| `GET` | `/api/reports/pending` | Get all `Under Review` reports (authority) |
| `GET` | `/api/reports/all` | Get all reports (authority archive) |
| `GET` | `/api/reports/{report_id}` | Get a single report |
| `PATCH` | `/api/reports/{report_id}/plate` | Update manual plate number |
| `PATCH` | `/api/reports/{report_id}/details` | Update location, description, violation type |

### Authority Actions

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/reports/{report_id}/approve` | Approve report (+150 karma to reporter) |
| `POST` | `/api/reports/{report_id}/reject` | Reject report |
| `GET` | `/api/authority/stats` | Get authority-wide statistics |

### Rewards

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/rewards/catalogue` | List available rewards |
| `POST` | `/api/rewards/redeem` | Redeem a reward (deducts karma points) |
| `GET` | `/api/rewards/history/{user_id}` | Get redemption history |

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server health check + AI model availability |

---

## Database Schema

The SQLite database (`backend/saral.db`) is auto-created on first run.

### `users`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `name` | TEXT | Display name |
| `email` | TEXT UNIQUE | Login email |
| `password` | TEXT | Plain-text password *(development only)* |
| `role` | TEXT | `user` or `authority` |
| `city` | TEXT | Optional city |
| `karma_points` | INTEGER | Accumulated karma points |
| `created_at` | TEXT | ISO timestamp |

### `reports`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | INTEGER FK | Reporter |
| `plate_number` | TEXT | OCR-detected plate |
| `manual_plate` | TEXT | Citizen-entered plate |
| `violation_type` | TEXT | e.g. `No Helmet`, `Traffic Violation` |
| `confidence` | REAL | AI confidence (0–100%) |
| `media_url` | TEXT | Path to original upload |
| `annotated_url` | TEXT | Path to annotated image |
| `location` | TEXT | Reported location |
| `description` | TEXT | Optional description |
| `status` | TEXT | `Under Review` / `Approved` / `Rejected` / `Auto-Rejected` |
| `helmet_detected` | TEXT | e.g. `2 with helmet, 1 without` |
| `created_at` | TEXT | ISO timestamp |

### `rewards_redeemed`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | INTEGER FK | Redeemer |
| `reward_id` | TEXT | e.g. `RW-001` |
| `title` | TEXT | Reward name |
| `cost` | INTEGER | Karma points spent |
| `redeemed_at` | TEXT | ISO timestamp |

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip
- A modern web browser

### 1. Clone the Repository
```sh
git clone https://github.com/Si-cyber-ai/SARAL.git
cd SARAL
```

### 2. Create a Virtual Environment (Recommended)
```sh
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python Dependencies
```sh
pip install -r requirements.txt
```

> **Note:** `easyocr` and `opencv-python` are large packages. The first install may take a few minutes. EasyOCR also downloads language model weights on its first use.

### 4. Start the Backend Server
```sh
cd backend
python main.py
```

The server starts at **`http://localhost:8000`**.  
Interactive API docs are available at **`http://localhost:8000/docs`**.

### 5. Open the Frontend
Open `index.html` in your browser (or navigate to `http://localhost:8000/static/index.html` to serve it through FastAPI's static file mount).

> The backend must be running for any API calls to work.

---

## Default Accounts

The database is seeded with two accounts on first run:

| Role | Name | Email | Password |
|---|---|---|---|
| Citizen | Aarav Kumar | `aarav@saral.in` | `citizen123` |
| Authority | Insp. T. Prasad | `prasad@authority.in` | `authority123` |

> ⚠️ **These are development credentials. Change them before any production deployment.**

---

## Rewards Catalogue

| ID | Reward | Cost (Karma Points) |
|---|---|---|
| RW-001 | FASTag Recharge (₹200 credit) | 150 |
| RW-002 | Metro Pass (5-day unlimited) | 120 |
| RW-003 | Fuel Voucher (₹300 discount) | 200 |
| RW-004 | Gift Card (₹500 Amazon/Flipkart) | 300 |

Citizens earn **150 karma points** per approved report. The rewards catalogue is defined in `backend/main.py` and can be extended easily.

### Karma Tiers

| Tier | Points Required |
|---|---|
| 🥉 Bronze | 0 – 499 |
| 🥈 Silver | 500 – 999 |
| 🥇 Gold | 1,000 – 1,499 |
| 💎 Platinum | 1,500 – 2,499 |
| 💠 Diamond | 2,500+ |

---

## Frontend Pages

| File | Role | Description |
|---|---|---|
| `index.html` | Public | Landing page with platform overview |
| `signin.html` | Public | Login for citizens and authorities |
| `signup.html` | Public | New citizen registration |
| `dashboard.html` | Citizen | Stats overview and quick report access |
| `report.html` | Citizen | Multi-step report submission with AI analysis |
| `my-reports.html` | Citizen | Full report history with status tracking |
| `rewards.html` | Citizen | Karma balance and reward redemption |
| `settings.html` | Citizen | Profile and preference management |
| `authority.html` | Authority | Pending report review queue |
| `authority-analytics.html` | Authority | Platform-wide statistics |
| `authority-archive.html` | Authority | Full report archive |
| `authority-settings.html` | Authority | Authority account settings |

### Frontend Architecture

The frontend uses three shared JavaScript modules loaded on every page:

- **`SaralAPI`** (`js/api.js`) — Centralized `fetch`-based API client. All HTTP calls go through this module.
- **`SaralAuth`** (`js/auth.js`) — Session management, role-based access control, and redirect helpers. Sessions are stored in `localStorage`.
- **`SaralStore`** (`js/store.js`) — Global reactive state store with `localStorage` persistence. Also exports `SaralToast` for in-app notifications.

---

## Configuration

### Backend Port
The server runs on port `8000` by default. To change it, edit the last line of `backend/main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Frontend API Base URL
The frontend API base URL is set in `js/api.js`:
```javascript
const BASE_URL = 'http://localhost:8000';
```
Update this if you deploy the backend to a different host or port.

### Roboflow API Key
The Roboflow API key and model IDs are configured at the top of `models/model.py`:
```python
ROBOFLOW_MODEL_ID = "indian-plate/1"
HELMET_MODEL_ID   = "helmet-detection-tiuol/1"
ROBOFLOW_API_KEY  = "your_api_key_here"
```

### Auto-Rejection Threshold
Reports with AI confidence below this percentage are automatically rejected. Defined in `backend/main.py`:
```python
auto_status = "Auto-Rejected" if confidence_pct < 30 else "Under Review"
```

### Karma Points Per Approval
```python
KARMA_POINTS_PER_APPROVAL = 150  # backend/main.py
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a branch** for your feature or bugfix: `git checkout -b feature/your-feature-name`
3. **Commit** your changes with clear messages.
4. **Push** to your fork and open a **Pull Request**.
5. For significant changes, please **open an issue first** to discuss your proposal.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Roboflow](https://roboflow.com/) — YOLOv8 model hosting and inference
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Open-source OCR library
- [OpenCV](https://opencv.org/) — Image processing
- Civic-tech communities across India for the inspiration behind SARAL
