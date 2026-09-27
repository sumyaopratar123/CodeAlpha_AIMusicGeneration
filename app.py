import os
import tempfile
import wave
import numpy as np
import streamlit as st

from music21 import converter, note, chord, stream, tempo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #191c24;
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .success-box {
        background-color: #123c27;
        padding: 16px;
        border-radius: 10px;
        color: #8ff0b2;
    }

    .error-box {
        background-color: #401c22;
        padding: 16px;
        border-radius: 10px;
        color: #ff9da7;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BUILT-IN MUSIC DATA
# ============================================================

STYLE_PATTERNS = {
    "Classical": [
        "C4", "D4", "E4", "F4",
        "G4", "A4", "G4", "F4",
        "E4", "D4", "C4", "E4",
        "G4", "C5", "B4", "A4",
        "G4", "E4", "F4", "D4"
    ],

    "Folk": [
        "C4", "C4", "G4", "A4",
        "G4", "E4", "D4", "C4",
        "E4", "G4", "A4", "G4",
        "E4", "D4", "C4", "C4",
        "D4", "E4", "G4", "E4"
    ],

    "Jazz": [
        "C4", "Eb4", "G4", "Bb4",
        "A4", "G4", "F4", "Eb4",
        "D4", "F4", "A4", "C5",
        "Bb4", "G4", "F4", "D4",
        "Eb4", "G4", "Bb4", "D5"
    ],

    "Pop": [
        "C4", "E4", "G4", "G4",
        "A4", "G4", "E4", "D4",
        "C4", "E4", "G4", "A4",
        "G4", "E4", "D4", "C4",
        "G4", "A4", "C5", "B4"
    ]
}


# ============================================================
# MIDI NOTE HELPERS
# ============================================================

def midi_number_to_frequency(midi_number):
    return 440.0 * (2.0 ** ((midi_number - 69) / 12.0))


def note_name_to_midi(name):
    try:
        n = note.Note(name)
        return int(n.pitch.midi)
    except Exception:
        return 60


def midi_to_note_name(midi_number):
    try:
        return note.Note(int(midi_number)).nameWithOctave
    except Exception:
        return "C4"


# ============================================================
# MIDI DATA EXTRACTION
# ============================================================

def extract_notes_from_midi(file_path):
    """
    Extract note names from a MIDI file using music21.
    Handles normal notes and chords.
    """

    extracted = []

    try:
        score = converter.parse(file_path)

        for element in score.flatten().notes:

            if isinstance(element, note.Note):
                extracted.append(element.nameWithOctave)

            elif isinstance(element, chord.Chord):
                # Use the highest note from a chord
                highest = element.pitches[-1]
                extracted.append(highest.nameWithOctave)

    except Exception as e:
        st.warning(f"Could not read MIDI file: {e}")

    return extracted


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data(note_names, sequence_length=12):
    """
    Converts notes into integer sequences for LSTM training.
    """

    if len(note_names) <= sequence_length:
        return None, None, None

    unique_notes = sorted(list(set(note_names)))

    note_to_int = {
        n: i for i, n in enumerate(unique_notes)
    }

    int_to_note = {
        i: n for n, i in note_to_int.items()
    }

    sequences = []
    targets = []

    for i in range(len(note_names) - sequence_length):
        sequence = note_names[i:i + sequence_length]
        target = note_names[i + sequence_length]

        sequences.append(
            [note_to_int[n] for n in sequence]
        )

        targets.append(
            note_to_int[target]
        )

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.int32)

    # Normalize inputs
    if len(unique_notes) > 1:
        X = X / float(len(unique_notes) - 1)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y, (note_to_int, int_to_note)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_lstm_model(vocab_size, sequence_length):

    model = Sequential([
        LSTM(
            128,
            input_shape=(sequence_length, 1),
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(128),

        Dropout(0.2),

        Dense(128, activation="relu"),

        Dense(
            vocab_size,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# GENERATE NOTES
# ============================================================

def generate_notes(
    model,
    seed_sequence,
    int_to_note,
    note_to_int,
    length,
    temperature=0.8
):

    generated = list(seed_sequence)

    sequence_length = len(seed_sequence)

    # Convert seed to integers
    current_sequence = [
        note_to_int[n]
        for n in seed_sequence
    ]

    for _ in range(length):

        x = np.array(
            current_sequence[-sequence_length:],
            dtype=np.float32
        )

        max_index = max(
            len(note_to_int) - 1,
            1
        )

        x = x / float(max_index)

        x = x.reshape(
            1,
            sequence_length,
            1
        )

        prediction = model.predict(
            x,
            verbose=0
        )[0]

        # Temperature sampling
        prediction = np.asarray(
            prediction
        ).astype(np.float64)

        prediction = np.log(
            prediction + 1e-8
        ) / max(
            temperature,
            0.05
        )

        probabilities = np.exp(prediction)

        probabilities /= probabilities.sum()

        next_index = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        next_note = int_to_note[int(next_index)]

        generated.append(next_note)

        current_sequence.append(
            int(next_index)
        )

    return generated


# ============================================================
# CREATE MIDI FILE
# ============================================================

def create_midi(notes_list, tempo_bpm=100):

    midi_stream = stream.Stream()

    midi_stream.append(
        tempo.MetronomeMark(
            number=tempo_bpm
        )
    )

    for note_name in notes_list:

        try:
            n = note.Note(note_name)

            n.quarterLength = 0.5

            midi_stream.append(n)

        except Exception:
            continue

    # IMPORTANT:
    # music21 requires an actual file path here.
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mid",
        delete=False
    )

    temp_path = temp_file.name
    temp_file.close()

    try:

        midi_stream.write(
            "midi",
            fp=temp_path
        )

        with open(
            temp_path,
            "rb"
        ) as f:

            midi_data = f.read()

        return midi_data

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# AUDIO SYNTHESIS
# ============================================================

def create_audio(
    notes_list,
    tempo_bpm=100,
    sample_rate=22050
):
    """
    Creates a simple WAV audio file directly using NumPy.
    No FluidSynth required.
    """

    seconds_per_beat = 60.0 / tempo_bpm

    note_duration = seconds_per_beat * 0.5

    audio_parts = []

    for note_name in notes_list:

        try:
            midi_num = note_name_to_midi(
                note_name
            )

            frequency = midi_number_to_frequency(
                midi_num
            )

            samples = int(
                sample_rate * note_duration
            )

            t = np.linspace(
                0,
                note_duration,
                samples,
                endpoint=False
            )

            # Basic musical tone with harmonics
            sound = (
                0.55 * np.sin(
                    2 * np.pi * frequency * t
                )
                +
                0.25 * np.sin(
                    2 * np.pi * frequency * 2 * t
                )
                +
                0.12 * np.sin(
                    2 * np.pi * frequency * 3 * t
                )
                +
                0.05 * np.sin(
                    2 * np.pi * frequency * 4 * t
                )
            )

            # ADSR-like envelope
            attack = max(
                1,
                int(samples * 0.05)
            )

            release = max(
                1,
                int(samples * 0.15)
            )

            envelope = np.ones(samples)

            envelope[:attack] = np.linspace(
                0,
                1,
                attack
            )

            envelope[-release:] = np.linspace(
                1,
                0,
                release
            )

            sound *= envelope

            audio_parts.append(sound)

        except Exception:
            continue

    if not audio_parts:
        return None

    audio = np.concatenate(
        audio_parts
    )

    # Normalize
    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:
        audio = audio / maximum

    audio = (
        audio * 32767 * 0.85
    ).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_path = temp_file.name
    temp_file.close()

    try:

        with wave.open(
            wav_path,
            "wb"
        ) as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(
                audio.tobytes()
            )

        with open(
            wav_path,
            "rb"
        ) as f:

            audio_data = f.read()

        return audio_data

    finally:

        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================
# SESSION STATE
# ============================================================

if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = []

if "midi_data" not in st.session_state:
    st.session_state.midi_data = None

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎵 AI Music Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate new melodies using an LSTM neural network trained on MIDI note sequences.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎛️ Music Settings")

    style = st.selectbox(
        "Music Style",
        [
            "Classical",
            "Folk",
            "Jazz",
            "Pop"
        ]
    )

    melody_length = st.slider(
        "Melody Length",
        min_value=16,
        max_value=128,
        value=64,
        step=8
    )

    temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )

    tempo_bpm = st.slider(
        "Tempo (BPM)",
        min_value=60,
        max_value=180,
        value=100,
        step=5
    )

    st.divider()

    st.info(
        "Upload MIDI files to train the AI "
        "on your own music dataset. "
        "If no files are uploaded, a built-in "
        "music corpus will be used."
    )

    uploaded_files = st.file_uploader(
        "Upload MIDI files",
        type=["mid", "midi"],
        accept_multiple_files=True
    )


# ============================================================
# MAIN
# ============================================================

col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("🎼 Generate Music")

    st.write(
        "The application trains a small LSTM network "
        "to learn note transitions and generate a new melody."
    )


with col2:

    st.metric(
        "Selected Length",
        melody_length
    )


# ============================================================
# TRAIN & GENERATE
# ============================================================

if st.button(
    "🎵 Train & Generate",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Preparing music dataset..."
    ):

        all_notes = []

        # ----------------------------------------------------
        # USER MIDI DATA
        # ----------------------------------------------------

        if uploaded_files:

            st.info(
                f"Processing {len(uploaded_files)} MIDI file(s)..."
            )

            for uploaded_file in uploaded_files:

                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".mid",
                    delete=False
                )

                temp_path = temp_file.name

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_file.close()

                try:

                    midi_notes = extract_notes_from_midi(
                        temp_path
                    )

                    all_notes.extend(
                        midi_notes
                    )

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        # ----------------------------------------------------
        # BUILT-IN DATA
        # ----------------------------------------------------

        if len(all_notes) < 30:

            all_notes = STYLE_PATTERNS[
                style
            ].copy()

            # Repeat corpus for training
            repetitions = max(
                10,
                200 // len(all_notes)
            )

            all_notes = (
                all_notes * repetitions
            )

            st.info(
                "Using built-in training music corpus."
            )

        else:

            st.success(
                f"Loaded {len(all_notes)} notes "
                "from uploaded MIDI data."
            )


    # ========================================================
    # PREPARE DATA
    # ========================================================

    sequence_length = 12

    X, y, mappings = prepare_training_data(
        all_notes,
        sequence_length
    )

    if X is None:

        st.error(
            "Not enough notes available for training."
        )

        st.stop()

    note_to_int, int_to_note = mappings

    vocab_size = len(note_to_int)


    # ========================================================
    # BUILD MODEL
    # ========================================================

    with st.spinner(
        "Building LSTM model..."
    ):

        tf.keras.backend.clear_session()

        model = build_lstm_model(
            vocab_size,
            sequence_length
        )


    # ========================================================
    # TRAIN
    # ========================================================

    progress = st.progress(
        0
    )

    status = st.empty()

    class ProgressCallback(
        tf.keras.callbacks.Callback
    ):

        def on_epoch_end(
            self,
            epoch,
            logs=None
        ):

            percent = int(
                ((epoch + 1) / epochs) * 100
            )

            progress.progress(
                min(percent, 100)
            )

            loss = logs.get(
                "loss",
                0
            )

            accuracy = logs.get(
                "accuracy",
                0
            )

            status.write(
                f"Training epoch "
                f"{epoch + 1}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )


    with st.spinner(
        "Training LSTM model..."
    ):

        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=min(
                32,
                len(X)
            ),
            verbose=0,
            callbacks=[
                ProgressCallback()
            ]
        )


    progress.progress(
        100
    )

    status.success(
        "LSTM training completed."
    )


    # ========================================================
    # GENERATE
    # ========================================================

    with st.spinner(
        "Generating new melody..."
    ):

        seed_length = min(
            sequence_length,
            len(all_notes)
        )

        seed_sequence = all_notes[
            :seed_length
        ]

        generated_notes = generate_notes(
            model=model,
            seed_sequence=seed_sequence,
            int_to_note=int_to_note,
            note_to_int=note_to_int,
            length=melody_length,
            temperature=temperature
        )

        # Remove seed from final output
        generated_notes = generated_notes[
            seed_length:
        ]

        if len(generated_notes) == 0:
            generated_notes = seed_sequence

        st.session_state.generated_notes = (
            generated_notes
        )


    # ========================================================
    # CREATE MIDI
    # ========================================================

    with st.spinner(
        "Creating MIDI file..."
    ):

        try:

            midi_data = create_midi(
                generated_notes,
                tempo_bpm
            )

            st.session_state.midi_data = (
                midi_data
            )

        except Exception as e:

            st.session_state.midi_data = None

            st.error(
                f"MIDI generation failed: {e}"
            )


    # ========================================================
    # CREATE AUDIO
    # ========================================================

    with st.spinner(
        "Creating audio..."
    ):

        try:

            audio_data = create_audio(
                generated_notes,
                tempo_bpm
            )

            st.session_state.audio_data = (
                audio_data
            )

        except Exception as e:

            st.session_state.audio_data = None

            st.error(
                f"Audio generation failed: {e}"
            )


    st.success(
        "Music generated successfully!"
    )


# ============================================================
# GENERATED MUSIC
# ============================================================

if st.session_state.generated_notes:

    st.divider()

    st.header("🎵 Generated Melody")

    notes_text = " ".join(
        st.session_state.generated_notes
    )

    st.code(
        notes_text,
        language="text"
    )


    # ========================================================
    # AUDIO PLAYER
    # ========================================================

    if st.session_state.audio_data:

        st.subheader(
            "🔊 Generated Audio"
        )

        st.audio(
            st.session_state.audio_data,
            format="audio/wav"
        )

        st.download_button(
            label="⬇️ Download Audio (WAV)",
            data=st.session_state.audio_data,
            file_name="ai_generated_music.wav",
            mime="audio/wav",
            use_container_width=True
        )

    else:

        st.warning(
            "Audio could not be generated."
        )


    # ========================================================
    # MIDI DOWNLOAD
    # ========================================================

    if st.session_state.midi_data:

        st.subheader(
            "🎹 MIDI File"
        )

        st.download_button(
            label="⬇️ Download MIDI",
            data=st.session_state.midi_data,
            file_name="ai_generated_music.mid",
            mime="audio/midi",
            use_container_width=True
        )


# ============================================================
# KNOWLEDGE BASE / INFO
# ============================================================

with st.expander(
    "📚 View Knowledge Base"
):

    st.markdown(
        """
        ### How this project works

        **1. MIDI Data Collection**
        - Upload MIDI files containing classical, folk,
          jazz or pop music.
        - The application extracts musical notes.

        **2. Preprocessing**
        - `music21` reads MIDI files.
        - Notes are converted into sequences.
        - Each sequence is used to predict the next note.

        **3. Deep Learning**
        - An LSTM-based RNN learns note transitions.
        - The model learns patterns from the training data.

        **4. Music Generation**
        - The trained model predicts new notes.
        - Temperature controls creativity/randomness.

        **5. MIDI Generation**
        - Generated notes are converted back into a MIDI file.

        **6. Audio Generation**
        - The generated melody is synthesized directly
          into a WAV audio file.
        - No external MIDI synthesizer is required.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Technology: Python • Streamlit • TensorFlow • "
    "LSTM • music21 • NumPy"
)