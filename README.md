# 🤖 AI-Powered CSV Data Analyzer

An intelligent Streamlit application for analyzing CSV/Excel data with AI-powered insights and automated business research capabilities.

## 🚀 Quick Deploy to Railway

This application is optimized for Railway deployment with built-in web scraping and business research capabilities.

## ✨ Features

- **Smart Data Loading:** Auto-detects identifier columns (HS codes, product codes)
- **AI Chat Interface:** Ask questions about your data in natural language
- **Multiple AI Providers:** Claude, Groq, OpenAI, and local analysis
- **Advanced Filtering:** Business intelligence focused data exploration
- **Automated Research:** Business contact finding with web scraping
- **Export Capabilities:** Enhanced datasets with research results

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run ai_csv_analyzer.py
```

## 🔐 Environment Variables

The app works without API keys but provides enhanced features when configured:

- `GROQ_API_KEY` - For Groq AI chat
- `OPENAI_API_KEY` - For OpenAI GPT chat  
- `ANTHROPIC_API_KEY` - For Claude AI chat
- `TAVILY_API_KEY` - For web scraping research

## 🚀 Railway Deployment

1. Fork this repository
2. Connect to Railway
3. Set environment variables in Railway dashboard
4. Deploy automatically

## 📊 Usage

1. Upload CSV or Excel files
2. Explore data with intelligent filters
3. Chat with your data using AI
4. Generate visualizations
5. Research business contacts (with API keys)
6. Export enhanced datasets

## 📝 License

MIT License - See LICENSE file for details.
