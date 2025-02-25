# Gyandex

## 🎯 Project Vision
Transform how we consume online content by providing flexible, AI-powered tools that convert text content into various formats. Currently focused on converting web articles and YouTube videos into podcast-ready audio content.

## ✨ Key Features
* Web article and YouTube video content extraction
* AI-powered conversion into natural-sounding podcast scripts
* High-quality text-to-speech synthesis with multiple voices
* Podcast feed generation for easy consumption
* Support for multiple content sources for a single podcast

## 🚀 Getting Started

### Prerequisites
* Python 3.11-3.13
* [Poetry](https://python-poetry.org/docs/#installation) for dependency management
* API keys for:
  * [Google AI](https://aistudio.google.com/app/apikey) (for content processing)
  * [Google Cloud](https://cloud.google.com/text-to-speech/docs/create-audio-text-client-libraries) (for text-to-speech)
  * S3-compatible storage (e.g., AWS S3, Cloudflare R2)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dhruvbaldawa/gyandex.git
   cd gyandex
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Copy `.env.example` to `.env` and set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your API keys:
   ```bash
   GOOGLE_API_KEY=your_google_ai_key
   GOOGLE_CLOUD_PROJECT=your_gcp_project
   ACCESS_KEY_ID=your_s3_access_key
   SECRET_ACCESS_KEY=your_s3_secret_key
   ```

### Configuration
Create a YAML configuration file (see `samples/` directory for examples) with:
* Content source (YouTube URL or web article)
* Workflow settings (AI providers and models)
* Text-to-speech voice configurations
* Storage settings
* Podcast feed metadata

### Usage
Generate a podcast from your configuration:
```bash
poetry run podgen your-config.yaml
```

## 📖 Documentation
Check the `samples/` directory for example configurations and common use cases.

## 🗺️ Roadmap
* Support for PDFs and other document formats
* Frontend for easy podcast creation and management
* Enhanced customization options for podcast creation
* Question-answering interface for content interaction

## License

This project is licensed under the **AGPL v3** for open-source use. For those wishing to use the software in proprietary applications without disclosing source code, a **commercial license** is available.

- **Open-Source License**: [AGPL v3](LICENSE)
- **Commercial License**: [Contact us](https://form.jotform.com/242954389750469) for more information.

By contributing to this repository, you agree that your contributions will be licensed under the same [AGPL v3 license](LICENSE).
