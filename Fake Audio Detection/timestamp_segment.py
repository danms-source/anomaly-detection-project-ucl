"""
Locate which chapter/segments a short clip belongs to
using anthem_transcriptions.json generated previously.
"""

import os, json, re, tempfile, shutil, warnings
from typing import List, Tuple

from whisper import load_model
from rapidfuzz import fuzz
import speech_recognition as sr
from pydub import AudioSegment

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
#  basic text helpers
# -----------------------------------------------------------

def normalise(txt: str) -> str:
    """lower-case, remove punctuation & double-spaces"""
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return re.sub(r"\s+", " ", txt).strip()

def concat_chapter_tokens(ch_data: dict) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Returns:
        big_string  (words separated by ' ')
        bounds      list[(start_word, end_word)] for every segment
    end_word is exclusive.
    """
    pieces = []
    bounds = []

    for seg in ch_data["segments"]:
        tokens = seg["text"].split()
        start = seg["starting_word_index"]
        end   = start + len(tokens)
        bounds.append((start, end))
        pieces.extend(tokens)

    return " ".join(pieces), bounds


# -----------------------------------------------------------
#  transcription of the query clip
# -----------------------------------------------------------

def transcribe_clip(path: str) -> str:
    """
    One-shot Google ASR; returns normalised text string.
    """
    # convert to mono/16k wav (Google works best)
    tmp_dir   = tempfile.mkdtemp(prefix="clip_")
    wav_path  = os.path.join(tmp_dir, "clip.wav")
    AudioSegment.from_file(path).set_channels(1).set_frame_rate(16_000)\
                .export(wav_path, format="wav")

    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as src:
            r.adjust_for_ambient_noise(src, duration=0.3)
            audio = r.record(src)
        text = r.recognize_google(audio, language="en-US")
    except (sr.UnknownValueError, sr.RequestError):
        text = ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return normalise(text)


# -----------------------------------------------------------
#  main search logic
# -----------------------------------------------------------

def find_best_match(query: str, data_json: str,
                    min_score: int = 70) -> dict:
    """
    Returns dict with chapter, segment list, start_time, end_time
    (or empty dict if nothing passes min_score).
    """
    with open(data_json, encoding="utf-8") as f:
        book = json.load(f)

    # 1. quick chapter scoring (token-set ratio against full chapter text)
    chap_scores = []
    for chap, ch_data in book.items():
        chap_text, _ = concat_chapter_tokens(ch_data)
        score = fuzz.token_set_ratio(query, chap_text)
        chap_scores.append((score, chap))
    chap_scores.sort(reverse=True)       # best first

    best_overall = {"score": 0}

    # 2. detailed sliding-window search
    for score, chap in chap_scores:
        ch_data = book[chap]
        full_text, bounds = concat_chapter_tokens(ch_data)
        words = full_text.split()
        q_len = max(3, len(query.split())) # avoid 0 length

        # build cumulative map: word_index -> segment_index
        seg_by_word = []
        for idx, (s, e) in enumerate(bounds):
            seg_by_word.extend([idx] * (e - s))

        # slide window over word indices; stride = 3 words for speed
        for start in range(0, len(words) - q_len + 1, 3):
            window_words = words[start:start + q_len + 50]  # +50 to allow fuzziness
            window_text  = " ".join(window_words)
            win_score = fuzz.token_set_ratio(query, window_text)
            if win_score > best_overall["score"]:
                first_seg = seg_by_word[start]
                last_seg  = seg_by_word[start + len(window_words) - 1]
                first_time = ch_data["segments"][first_seg]["starting_time_seconds"]
                # end time = start of seg after last_seg (or chap duration)
                if last_seg + 1 < len(ch_data["segments"]):
                    end_time = ch_data["segments"][last_seg + 1]["starting_time_seconds"]
                else:
                    end_time = ch_data["total_duration_seconds"]

                best_overall = {
                    "score": win_score,
                    "chapter": chap,
                    "matched_segments": list(range(first_seg, last_seg + 1)),
                    "start_time": round(first_time, 2),
                    "end_time":   round(end_time,   2)
                }

    return best_overall if best_overall["score"] >= min_score else {}

def transcribe_audio_segment_pydub(file_path, start_time, end_time, model):
    """
    Transcribe a specific segment of an MP3 file using pydub.
    
    Args:
        file_path (str): Path to the MP3 file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        model (Whisper): Whisper model
    
    Returns:
        WhisperResult: Transcription results
    """
    
    # Load the audio file
    audio = AudioSegment.from_mp3(file_path)
    
    # Extract the segment (pydub works in milliseconds)
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    segment = audio[start_ms:end_ms]
    
    # Create a temporary file for the audio segment
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Export the segment
        segment.export(temp_path, format="mp3")
        
        # Transcribe the segment
        results = model.transcribe(temp_path)
        
        # Adjust timestamps to reflect the original position in the full audio
        # Access segments using the .segments attribute
        for segment in results["segments"]:
            segment["start"] += start_time
            segment["end"] += start_time
            
            # Adjust word timestamps if they exist
            if 'words' in segment:
                for word in segment["words"]:
                    word["start"] += start_time
                    word["end"] += start_time
            
            # Adjust whole_word_timestamps if they exist
            if 'whole_word_timestamps' in segment:
                for word_ts in segment["whole_word_timestamps"]:
                    word_ts["timestamp"] += start_time
        
        return results
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------------------------------------
#  driver
# -----------------------------------------------------------

def main():
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(base_dir, 'anthem_transcriptions.json')
    if not os.path.isfile(json_path):
        print("✖ JSON not found")
        return
    
    audio_dir = os.path.join(base_dir, 'anthem_1308_librivox')
    if not os.path.isdir(audio_dir):
        print("✖ Audio directory not found")

    clip_path = os.path.join(base_dir, 'fake_clips', 'Chap_1', 'tmpougjqlfz.wav') # clip to determine if fake or not
    if not os.path.isfile(clip_path):
        print("✖ Clip not found")
        return
    
    model_size = "tiny.en"
    model = load_model(model_size)

    print("\nTranscribing clip …")
    query_text = transcribe_clip(clip_path)
    if not query_text:
        print("✖ Couldn’t transcribe the clip")
        return
    print(f"✓ Clip transcript (normalised):\n  \"{query_text}\"\n")

    print("Searching best match …")
    match = find_best_match(query_text, json_path)

    if not match:
        print("No good match found (score < 70)")
        print(json.dumps(match, indent=2))
    else:
        print("Found good match")
        print(json.dumps(match, indent=2))
        results = transcribe_audio_segment_pydub(
            file_path=os.path.join(audio_dir, f"anthem_{int(match["chapter"][-2:]):02d}_rand_64kb.mp3"),
            start_time=match["start_time"],
            end_time=match["end_time"],
            model=model,
        )

        # Print the transcribed text
        print(f"Transcription from {match["start_time"]}s to {match["end_time"]}s:")
        # Access segments using the .segments attribute
        for segment in results["segments"]:
            score = fuzz.token_set_ratio(normalise(segment["text"]), query_text)
            if score > 90:
                print(f"\033[92m [{segment["start"]:.2f}s - {segment["end"]:.2f}s] {segment["text"]}\033[00m")
            elif score > 80:
                print(f"\033[93m [{segment["start"]:.2f}s - {segment["end"]:.2f}s] {segment["text"]}\033[00m")
            else:
                print(f"[{segment["start"]:.2f}s - {segment["end"]:.2f}s] {segment["text"]}")

if __name__ == "__main__":
    main()