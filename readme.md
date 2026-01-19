
## Inspiration

The inspiration for this project came from the persistent communication barriers faced by Deaf, hard-of-hearing, and speech-impaired individuals in a world still largely designed for hearing and speaking users. We wanted to explore how technology could move beyond captions and create truly inclusive, real-time communication—while also making sign language learning accessible, engaging, and intuitive.

## What it does

This project is a **working prototype** that translates English text or speech into simplified, sign-language-compatible keywords and retrieves corresponding sign language video demonstrations. It also features **real-time gesture recognition**, allowing users to practice signs and receive immediate visual feedback.

The prototype demonstrates the foundations of a **multimodal communication system**, bridging text, speech, and sign language in an accessible and interactive way.

## How to run (development & local) 🔧

Follow these steps to run the app locally on Windows (PowerShell). The instructions cover both the default `config.toml` setup and an environment variable option.

### 1) Prerequisites

- Python 3.9+ (tested on Python 3.13). Use a virtual environment.
- A webcam and microphone for camera/speech features (optional).

### 2) Prepare the virtual environment (Windows PowerShell)

```powershell
# create venv
python -m venv .venv
# activate (PowerShell)
& ".\.venv\Scripts\Activate.ps1"
# install dependencies
pip install -r requirements.txt
# (or if you don't have requirements.txt)
pip install streamlit opencv-python mediapipe groq pillow requests numpy
```

> Tip: If you get errors installing `mediapipe`, try upgrading pip and retry: `python -m pip install --upgrade pip` then `pip install mediapipe`.

### 3) Configure your Groq API key (two options)

Option A — `config.toml` (default):

Create a `config.toml` in the project root with this content (do not check it into git):

```toml
[api]
key = "your_groq_api_key_here"
```

Option B — Environment variable (optional, more secure):

```powershell
# Session scope
$env:GROQ_API_KEY = "your_groq_api_key_here"
# Persist across sessions (Windows)
setx GROQ_API_KEY "your_groq_api_key_here"
```

Note: The app currently reads `config.toml`. To prefer an environment variable first, update `userInterface.py` to check `os.getenv('GROQ_API_KEY')` before loading `config.toml`.

### 4) Start the app

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run userInterface.py
```

Open your browser at `http://localhost:8501` (Streamlit prints the exact URL and port in the terminal).

### 5) Permissions & runtime notes

- Allow your browser to use the camera and microphone for those features.
- If your webcam is used by another application, the camera feature may fail.
- If the installed MediaPipe does not expose `mp.solutions`, the Camera Translation feature will be disabled by the app with an explanatory message; you can either install a compatible MediaPipe or update the app to use MediaPipe **Tasks** (the app already has a safe fallback).

### 6) Quick feature tests

- **Text Translation**: Enter text into the text box and click **Convert to BSL Videos**.
- **Speech Translation**: Click the microphone, record, then click **Transcribe & Generate BSL Videos**.
- **Camera Translation**: Click **Start Real-time Gesture Detection** (if available on your MediaPipe build).

### 7) Troubleshooting (common errors)

- ModuleNotFoundError: install the missing package, e.g. `pip install mediapipe` or `pip install groq`.
- cv2 (OpenCV) missing: `pip install opencv-python`.
- Groq authentication errors: verify `config.toml` or `GROQ_API_KEY` and your network connectivity.
- `logo.png` not found: place a `logo.png` in the project root to show the sidebar logo.

---

## How we built it

We built the prototype using **Streamlit** for the interactive interface, **MediaPipe** for real-time hand and gesture detection, and the **Groq AI API** to simplify English input into sign-language-friendly keywords. Sign demonstration videos were sourced from **SignBSL**, with fallback mechanisms implemented for cases where specific signs were unavailable. **OpenCV** and **PIL** were used for image and video processing to ensure smooth real-time interaction.

## Challenges we ran into

One of the main challenges was ensuring accurate alignment between English words and sign language keywords, particularly when direct equivalents did not exist. Managing missing sign resources, maintaining low-latency gesture recognition, and designing a system that remains intuitive for both learners and native signers were also significant challenges.

## Accomplishments that we're proud of

* A functional, end-to-end prototype demonstrating text/speech → sign workflows
* Real-time gesture recognition with immediate user feedback
* Successful integration of generative AI with accessibility-focused design
* A strong proof of concept for a broader multimodal communication platform

## What we learned

We learned that sign languages have deep grammatical and cultural nuances that cannot be treated as direct translations of spoken language. We also gained valuable insights into gesture recognition, accessibility-first UX design, and the importance of building assistive technologies **with inclusivity at the core, not as an afterthought**.

## What’s next

This prototype is the foundation for **Arkan Infinity**, and the next phase will significantly expand its technical and accessibility capabilities.

We plan to integrate **OpenPose alongside MediaPipe** to enable more precise **full upper-body motion capture**, improving recognition accuracy and expressiveness. To enhance realism and comprehension, we will incorporate **photorealistic avatars** using technologies such as **Unreal Engine MetaHuman** or **Higgsfield AI** for natural sign language rendering.

On the hardware side, we will continue developing **ModuSign Mini**, a wearable gesture-tracking device designed to enable camera-independent interaction and greater mobility.

Beyond sign language, we aim to extend our motion-capture technology to deliver **advanced accessibility tools** for people with disabilities, including individuals with reduced mobility, paraplegia, or limb loss. By leveraging alternative input signals—such as **facial muscle movements**—we envision enabling meaningful interaction with digital interfaces even when traditional input methods are not possible.

Together, these steps will transform the project from an educational prototype into a **universal multimodal communication and accessibility platform**.

## Built With

* Streamlit – interactive interface
* MediaPipe – real-time gesture detection
* Groq AI API – text simplification (Generative AI)
* SignBSL.com – sign language video resources
* OpenCV & PIL – image and video processing

