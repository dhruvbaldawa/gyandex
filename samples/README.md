# Sample Configurations

This directory contains example configurations for different use cases of Gyandex.

## Available Examples

### youtube-to-podcast.yaml
A basic configuration that demonstrates how to convert a YouTube video into a podcast episode. This example shows:
- YouTube video source configuration
- AI processing settings for content transformation
- Text-to-speech voice configuration with host and expert voices
- Storage settings
- Podcast feed metadata

### website-to-podcast.yaml
A configuration template for converting web articles into podcast episodes. This example demonstrates:
- Web article source configuration
- AI processing for article content
- Text-to-speech setup with narrator and analyst voices
- Storage configuration
- Podcast feed settings for article-based content

## Configuration Options

### Content Source
```yaml
content:
  source: "URL"  # URL of the content (YouTube, web article, etc.)
  format: "youtube"  # Format: youtube, html, etc.
```

### Workflow
```yaml
workflow:
  name: "alexandria"  # Workflow type
  verbose: true      # Enable detailed logging
  outline:           # AI settings for content outline
    provider: "google-generative-ai"
    model: "gemini-2.0-flash-thinking-exp-01-21"
    temperature: 0.4
    api_key: "${GOOGLE_API_KEY}"
  script:            # AI settings for script generation
    provider: "google-generative-ai"
    model: "gemini-2.0-flash"
    temperature: 0.8
    api_key: "${GOOGLE_API_KEY}"
```

### Text-to-Speech
```yaml
tts:
  provider: "google-cloud"
  participants:
    - name: "Host"        # Character name
      personality: |      # Character personality description
        Description of the character's style and traits
      voice: "voice-id"   # TTS voice identifier
      language_code: "en-US"
      gender: "female"
```

### Storage
```yaml
storage:
  provider: "s3"     # S3-compatible storage
  access_key: "${ACCESS_KEY_ID}"
  secret_key: "${SECRET_ACCESS_KEY}"
  bucket: "bucket-name"
  region: "region"
  endpoint: "endpoint-url"  # Optional, for S3-compatible storage
```

### Podcast Feed
```yaml
feed:
  title: "Podcast Title"
  slug: "podcast-slug"     # URL-friendly identifier
  description: "Podcast description"
  author: "Author Name"
  email: "contact@example.com"
  language: "en"
  categories: ["Category1", "Category2"]
  image: "cover-image-url"
  website: "podcast-website-url"
```

## Environment Variables
Make sure to set these environment variables in your `.env` file:
- `GOOGLE_API_KEY`: Google AI API key
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `ACCESS_KEY_ID`: S3 access key
- `SECRET_ACCESS_KEY`: S3 secret key

## Usage

1. Copy the example configuration and modify it for your needs
2. Replace placeholder values with your actual configuration
3. Run using uvx (recommended):
   ```bash
   uvx run podgen your-config.yaml
   ```

   Or if you've installed manually using Poetry:
   ```bash
   poetry run podgen your-config.yaml
   ```
