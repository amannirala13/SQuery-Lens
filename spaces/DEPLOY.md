# Hugging Face Space Deployment

This directory contains the Gradio app for deploying SQuery-Lens to Hugging Face Spaces.

## 🚀 Live Demo

**URL**: https://huggingface.co/spaces/amannirala13/squery-lens-demo

Try the model directly in your browser!

## 📁 Files

- `app.py` - Gradio interface for Query Analyzer and Schema Ranker
- `requirements.txt` - Python dependencies
- `README.md` - Space metadata and description
- `src/` - Source code (copied from parent)

## 🔄 Update the Space

To update the Space after making changes:

```bash
# 1. Update source code
cp -r ../src ./

# 2. Upload to Space
source ../venv/bin/activate
huggingface-cli upload amannirala13/squery-lens-demo . --repo-type space
```

## 🛠️ Test Locally

```bash
cd spaces
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860 in your browser.

## 📝 Notes

- Models are downloaded automatically from `amannirala13/squery-lens-models` on startup
- The Space uses Gradio SDK version 4.44.0
- Free tier has 2 vCPU, 16GB RAM (sufficient for inference)
