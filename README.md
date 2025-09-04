# Medicaid Audit Report Analyzer

A comprehensive Flask-based web application for processing, analyzing, and managing government audit reports with AI-powered data extraction and intelligent document classification.

## 📋 Overview

The Medicaid Audit Report Analyzer is an administrative tool designed to streamline the processing of PDF audit reports from various government agencies. It combines advanced PDF parsing techniques with AI-powered analysis to extract structured data, identify key findings and recommendations, and manage report metadata efficiently.

## ✨ Key Features

### 📄 Document Processing
- **In-memory PDF processing** for secure handling of sensitive documents
- **Multiple parsing strategies** including PyMuPDF and Unstructured.io integration
- **Automatic duplicate detection** using SHA-256 hashing
- **Batch upload support** for processing multiple reports simultaneously

### 🤖 AI-Powered Analysis
- **Structured data extraction** using OpenAI GPT and Google Gemini models
- **Automatic classification** of audit documents
- **Entity extraction** for findings, recommendations, and objectives
- **Cost tracking** with actual token usage monitoring
- **Smart fallback** between AI providers for reliability

### 🔍 Search & Discovery
- **Automated audit discovery** using Google Custom Search API
- **AI-powered document classification** to identify relevant reports
- **Queue management system** with review workflow
- **Duplicate detection** across reports and processing queue

### 📊 Data Management
- **Comprehensive keyword mapping** system with normalization
- **SEO-friendly URL slugs** for public-facing content
- **Report relationship tracking** between findings and recommendations
- **Statistical analysis** of extracted data

### 🛠 Administrative Tools
- **Dashboard** with real-time statistics and recent activity
- **Keyword mapping interface** for terminology standardization
- **Comparison tools** for evaluating parsing strategies
- **Text chunking analysis** with multiple strategies
- **Report editing interface** for manual corrections

## 🏗 Technical Architecture

### Backend Stack
- **Flask** - Core web framework
- **SQLAlchemy** - ORM for database operations
- **PostgreSQL** - Primary database
- **Gunicorn** - WSGI HTTP server

### AI & NLP
- **OpenAI API** - GPT models for text extraction
- **Google Gemini API** - Alternative AI provider
- **Instructor** - Structured output parsing
- **LlamaIndex** - Semantic text processing
- **tiktoken** - Token counting and cost estimation

### Document Processing
- **PyMuPDF** - PDF text extraction
- **Unstructured.io** - Advanced document parsing
- **SHA-256** - Document fingerprinting

### Frontend
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome** - Icon library
- **Dark theme** - Professional admin interface

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL database
- API keys for:
  - OpenAI API
  - Google Gemini API
  - Google Custom Search API (optional)

### Environment Variables

Create a `.env` file with the following variables:

```bash
DATABASE_URL=postgresql://username:password@localhost/dbname
SESSION_SECRET=your-session-secret
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
GOOGLE_CSE_API_KEY=your-google-search-key (optional)
GOOGLE_CSE_ENGINE_ID=your-search-engine-id (optional)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medicaid-audit-analyzer.git
cd medicaid-audit-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

4. Run the application:
```bash
gunicorn main:app --bind 0.0.0.0:5000
```

## 📱 Usage

### Uploading Reports

1. Navigate to the upload interface (`/upload`)
2. Select one or more PDF files
3. Choose AI provider (OpenAI or Gemini)
4. Review extracted data
5. Confirm and save to database

### Managing Keywords

1. Access keyword mapping dashboard (`/mapping-review`)
2. Review unmatched keywords
3. Create canonical mappings for terminology variations
4. Monitor keyword usage across reports

### Automated Search

1. Configure search parameters (`/audit-search`)
2. Review AI-classified results
3. Approve reports for processing
4. Monitor queue status

## 🔄 Workflow

```
PDF Upload → Text Extraction → AI Analysis → Data Review → Database Storage
                     ↓                              ↓
              Duplicate Check              Keyword Extraction
                                                    ↓
                                           Mapping & Normalization
```

## 📈 API Endpoints

- `GET /api/popular-keywords` - Returns normalized keywords with report counts
- `GET /api/reports` - Paginated report listing
- `POST /api/upload` - Report upload endpoint
- `GET /api/queue/status` - Processing queue status

## 🛡 Security Features

- In-memory PDF processing (no disk storage)
- SHA-256 hash verification
- Session-based authentication
- SQL injection protection via SQLAlchemy ORM
- Environment-based configuration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Flask and SQLAlchemy
- AI capabilities powered by OpenAI and Google Gemini
- PDF processing via PyMuPDF
- UI framework by Bootstrap

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This application is designed for administrative use and requires appropriate API credentials for full functionality.