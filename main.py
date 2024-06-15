import argparse
import torch
from TTS.api import TTS
from tqdm import tqdm
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Convert text to speech using Coqui TTS")
    parser.add_argument('--text_file', type=str, required=True, help="Path to the text file containing the text to be converted.")
    parser.add_argument('--output_path', type=str, required=True, help="Path where the generated audio file will be saved.")
    parser.add_argument('--speed', type=float, default=1.0, help="Speech speed (e.g., 1.0 for normal speed).")
    parser.add_argument('--language', type=str, default="pt", help="Language of the text (e.g., 'pt' for Portuguese).")
    parser.add_argument('--sample_voice', type=str, required=True, help="Path to the audio file with the sample voice for cloning.")
    return parser.parse_args()

def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += (word + " ")
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main():
    args = parse_args()

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # Load text from file
    with open(args.text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Replace periods with commas
    text = text.replace('.', ',')

    # Split text into chunks
    text_chunks = split_text(text)

    # Initialize TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    # Temporary directory for storing chunk audio files
    temp_dir = "temp_audio_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    chunk_files = []

    start_time = time.time()

    # Convert each chunk to speech
    for idx, chunk in enumerate(tqdm(text_chunks, desc="Generating audio chunks")):
        chunk_file = os.path.join(temp_dir, f"chunk_{idx}.wav")
        tts.tts_to_file(
            text=chunk,
            speaker_wav=args.sample_voice,
            file_path=chunk_file,
            speed=args.speed,
            language=args.language
        )
        chunk_files.append(chunk_file)

    # Concatenate all chunks into the final output file
    concatenate_audio_files(chunk_files, args.output_path)

    # Clean up temporary files
    for file in chunk_files:
        os.remove(file)
    os.rmdir(temp_dir)

    end_time = time.time()
    total_time = end_time - start_time
    total_words = len(text.split())

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total words narrated: {total_words}")

def concatenate_audio_files(audio_files, output_path):
    import wave

    with wave.open(output_path, 'wb') as outfile:
        for i, infile_path in enumerate(audio_files):
            with wave.open(infile_path, 'rb') as infile:
                if i == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))

if __name__ == "__main__":
    main()
