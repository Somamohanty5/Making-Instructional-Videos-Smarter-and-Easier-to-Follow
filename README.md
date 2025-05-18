# Making-Instructional-Videos-Smarter-and-Easier-to-Follow
# Instruction Matching in Instructional YouTube Videos

This project presents an AI-powered system that aligns narration from instructional YouTube videos with their corresponding step-by-step instructions using both NLP and semantic AI techniques. The system transcribes audio, extracts instructional sentences, and matches them with structured instructions using two methods: word-overlap-based (NLP) and semantic similarity-based (AI). It further identifies the action, tool, and purpose of each instruction.

---

## Features

* **Audio Transcription**: Uses Whisper ASR to transcribe speech from YouTube video audio.
* **Instruction Extraction**: Extracts likely instructional sentences from transcripts using POS tagging and heuristics.
* **NLP-Based Matching**: Matches instructions using token overlap and inverse word frequency weighting.
* **AI-Based Matching**: Matches using Sentence-BERT embeddings and cosine similarity.
* **Feature Extraction**: Extracts Action, Tool, and Purpose using spaCy and AllenNLP SRL.
* **Evaluation & Visualization**: Computes precision, recall, F1 for NLP vs AI matches; visualizes results with boxplots and word clouds.

---

## Problem Statement

YouTube is widely used for learning, but long, unstructured videos and messy transcripts make it hard to find or verify step-by-step instructions. Creators struggle to ensure their narration matches their intended steps. This project solves these issues by building an automated pipeline that:

* Extracts instruction-like lines from transcript
* Matches them with known steps
* Evaluates how well the video narration reflects the structured guidance

---

## Dataset

**Source**: GitHub - Instructional-Video-Summarization

* Each JSON file represents a single instructional video
* Fields: `video_url`, `text` (steps), `img`, `vid`
* \~2000 videos from domains like Cooking, DIY, and Self-Care
* All JSON files are stored in the `how_to_steps/` folder

---

## Methodology

### Data Preparation

* Parse JSON files to extract YouTube link and reference steps
* Download and preprocess audio using `yt-dlp` and `ffmpeg`
* Transcribe audio with **Whisper** or fallback to **YouTubeTranscriptAPI**

### Instruction Extraction

* Split transcript into sentences using NLTK
* Filter instructional lines via POS tagging (keep verbs like "fold", "cut")
* Deduplicate similar lines using SentenceTransformer-based cosine similarity

### Instruction Matching

* **NLP Matching**: Word overlap + inverse frequency scoring to find best matching sentence
* **AI Matching**: SentenceTransformer (`all-MiniLM-L6-v2`) used to compute embeddings and match semantically

### Feature Extraction

* **Action**: Extracted using dependency parsing (main verb)
* **Tool**: Extracted as direct object or fallback via NER
* **Purpose**: Extracted using AllenNLP Semantic Role Labeling (ARGM-PURP or ARG1)

### Evaluation

* Compute **Precision, Recall, F1** between matched transcript steps and reference JSON
* Store in CSV: `comparison_scores.csv` and `steps_table.csv`
* Visualize results: boxplots, cosine similarity charts, word clouds

---

## Key Results

* AI-based matching (semantic) outperformed NLP-based matching (word-based) in terms of median F1 score
* Word clouds showed most mismatches happen on common instructional terms ("use", "make", "you")
* AI showed stronger performance for both short and long instructions
* Extracted tables show clear Action, Tool, and Purpose per step, aiding instructional clarity

---

### Output Files

* `steps_table.csv`: Instructions with action/tool/purpose and NLP/AI match
* `comparison_scores.csv`: NLP vs AI matching performance per video

---

## Dependencies

* Python 3.8+
* Whisper
* yt-dlp
* SentenceTransformer
* spaCy (`en_core_web_sm`)
* AllenNLP + SRL model
* scikit-learn, pandas, matplotlib, nltk, re

See `requirements.txt`[View the Requirements](requirements.txt) for full list.

---

## Future Plans

* Remove dependency on JSON: support URL-only input
* Enhance matching with BERTScore or instruction-tuned LLMs
* Train custom NER for instructional domains
* Deploy as web-based tool or command-line utility

---

## License

This project is released under the MIT License. See `LICENSE` for more info.
