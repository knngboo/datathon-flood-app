# Datathon flood app

Team project for the Dell Social Impact AI Datathon (UTSA, Nov 13, 2026).

The AI (Qwen3-VL-8B, a model that reads photos) runs on the team's Dell GB10. This app sends flood photos and questions to it and shows the answers.

## Run it on the Dell
```
python -m streamlit run app.py --server.address 0.0.0.0
```
Then open `http://<dell address>:8501`. The AI must be running at `http://localhost:8000/v1`.

`DECISIONS.md` has what we decided and why.
