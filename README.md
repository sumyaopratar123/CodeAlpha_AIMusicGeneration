# Task 3 — AI Music Generation

This project uses a small PyTorch LSTM to learn next-note patterns from an included training corpus. It then samples a new note sequence and exports it as MIDI.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first training run is CPU-friendly but may take a little time. The generated MIDI can be opened in a MIDI player or DAW.
