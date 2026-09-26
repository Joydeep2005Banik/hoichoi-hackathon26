# Hoichoi P4 — Creative Reformatting Engine

Subject-aware image & video reformatting for multiple aspect ratios.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Does
- **Image** → 16:9, 1:1, 4:5, 9:16 crops anchored on detected faces/subjects.
- **Video** → 9:16 vertical reel with dynamic crop following the active speaker.
- Every output is machine-validated before appearing in the library.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full details.
