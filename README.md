# Study Guide Generator

Generate comprehensive study guides from academic documents using AI.

## Requirements

- Python 3.11
- NVIDIA API access or OpenAI API key
- LlamaParse API key (for PDF parsing)

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/StudyGuideGenerator.git
   cd StudyGuideGenerator
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
    # OpenAI API Model
    OPENAI_MODEL=

    # For frontend
    API_URL=http://localhost:8000

    # Model from nvidia library
    NVIDIA_MODEL=

    # Embedding Model
    EMBED_MODEL=

    # LlamaIndex (LlamaCloud) API key
    LLAMA_CLOUD_API_KEY=

    # NVIDIA API key (must start with “nvapi-”)
    NVIDIA_API_KEY=

    # OpenAI API key
    OPENAI_API_KEY=
   ```

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

## Using the Study Guide Generator

1. Upload an academic PDF document
2. Enter a study topic that's covered in your document
3. Click "Generate Study Guide"
4. Wait while the AI analyzes your document and creates a study guide
5. Download the generated study guide as Markdown or text

## Features

- Creates structured study guides with key questions and answers
- Includes review questions for self-assessment
- Provides key terms and definitions
- Offers study tips related to the topic
- Cites sources from your document