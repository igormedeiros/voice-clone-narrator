import argparse
import torch
from TTS.api import TTS
from tqdm import tqdm
import os
import time
import nltk
import shutil
import logging
from datetime import datetime
from moviepy.editor import AudioFileClip, ImageClip

# Download the necessary NLTK data
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="Convert text to speech using Coqui TTS")
    parser.add_argument('--text_file', type=str, required=True, help="Path to the text file containing the text to be converted.")
    parser.add_argument('--output_path', type=str, required=True, help="Path where the generated audio file will be saved.")
    parser.add_argument('--speed', type=float, default=1.0, help="Speech speed (e.g., 1.0 for normal speed).")
    parser.add_argument('--language', type=str, default="pt", help="Language of the text (e.g., 'pt' for Portuguese).")
    parser.add_argument('--sample_voice', type=str, required=True, help="Path to the audio file with the sample voice for cloning.")
    parser.add_argument('--thumb', type=str, help="Path to the thumbnail image for MP4 output.")
    return parser.parse_args()

def split_text_into_sentences(text):
    # Replace symbols with placeholders for pauses
    text = text.replace('##', ' <PAUSE_2> ').replace('#', ' <PAUSE_1> ')
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Further split sentences that exceed the character limit
    max_chunk_size = 203
    chunks = []
    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            words = sentence.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                    current_chunk += (word + " ")
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = word + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks.append(sentence)
    return chunks

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}h{int(mins):02d}min{int(secs):02d}sec"

def main():
    args = parse_args()

    # Setup logging
    log_filename = f"narrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info("Starting the text-to-speech conversion process")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")

    # Load text from file
    with open(args.text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into sentences
    sentences = split_text_into_sentences(text)

    # Initialize TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    # Temporary directory for storing chunk audio files
    temp_dir = "temp_audio_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    chunk_files = []

    start_time = time.time()

    # Convert each sentence to speech and add pauses
    for idx, sentence in enumerate(tqdm(sentences, desc="Generating audio chunks")):
        parts = sentence.split()
        current_sentence = ""
        for part in parts:
            if part == "<PAUSE_1>":
                if current_sentence:
                    chunk_files.append(generate_audio_chunk(tts, current_sentence, args, temp_dir, len(chunk_files)))
                    current_sentence = ""
                add_pause(chunk_files, 1, temp_dir, len(chunk_files))
            elif part == "<PAUSE_2>":
                if current_sentence:
                    chunk_files.append(generate_audio_chunk(tts, current_sentence, args, temp_dir, len(chunk_files)))
                    current_sentence = ""
                add_pause(chunk_files, 2, temp_dir, len(chunk_files))
            else:
                current_sentence += part + " "
        if current_sentence:
            chunk_files.append(generate_audio_chunk(tts, current_sentence.strip(), args, temp_dir, len(chunk_files)))

    # Concatenate all chunks into the final output file
    audio_output_path = args.output_path
    concatenate_audio_files(chunk_files, audio_output_path)

    # Clean up temporary files
    for file in chunk_files:
        os.remove(file)
    
    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    if args.thumb:
        video_output_path = args.output_path.replace('.wav', '.mp4')
        generate_video_with_thumbnail(audio_output_path, args.thumb, video_output_path)
        os.remove(audio_output_path)
        logging.info(f"Generated video output at {video_output_path}")

    end_time = time.time()
    total_time = end_time - start_time
    total_words = len(text.split())
    words_per_minute = total_words / (total_time / 60)

    logging.info(f"Total time: {format_time(total_time)}")
    logging.info(f"Total words narrated: {total_words} ({words_per_minute:.0f} words per minute)")

def generate_audio_chunk(tts, text, args, temp_dir, idx):
    chunk_file = os.path.join(temp_dir, f"chunk_{idx}.wav")
    tts.tts_to_file(
        text=text,
        speaker_wav=args.sample_voice,
        file_path=chunk_file,
        speed=args.speed,
        language=args.language
    )
    return chunk_file

def add_pause(chunk_files, duration, temp_dir, idx):
    silence_file = os.path.join(temp_dir, f"silence_{duration}s_{len(chunk_files)}.wav")
    generate_silence(silence_file, duration)
    chunk_files.append(silence_file)

def generate_silence(file_path, duration):
    import wave
    import numpy as np

    sample_rate = 22050  # Sample rate of 22.05 kHz
    num_samples = duration * sample_rate
    silence = np.zeros(num_samples, dtype=np.int16)

    with wave.open(file_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(silence.tobytes())

def concatenate_audio_files(audio_files, output_path):
    import wave

    with wave.open(output_path, 'wb') as outfile:
        for i, infile_path in enumerate(audio_files):
            with wave.open(infile_path, 'rb') as infile:
                if i == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))

def generate_video_with_thumbnail(audio_path, thumbnail_path, video_path):
    audio_clip = AudioFileClip(audio_path)
    image_clip = ImageClip(thumbnail_path, duration=audio_clip.duration)
    video = image_clip.set_audio(audio_clip)
    video.write_videofile(video_path, codec='libx264', fps=24)

if __name__ == "__main__":
    main()
