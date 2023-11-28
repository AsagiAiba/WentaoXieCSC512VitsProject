import whisper
import os
import json
import torchaudio
import argparse
import torch

def load_whisper_model(whisper_size):
    assert torch.cuda.is_available(), "Check GPU"
    return whisper.load_model(whisper_size)

def is_english(probs):
    return 'en' in probs

def process_audio_file(audio_path, target_sr):
    try:
        wav, sr = torchaudio.load(audio_path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
        wav = wav.mean(dim=0).unsqueeze(0)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        if wav.shape[1] / sr > 20:
            print(f"{audio_path} is too long, ignoring\n")
            return None, None
        return wav, sr
    except:
        print(f"Failed to process {audio_path}")
        return None, None

def transcribe(model, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return lang, result.text

def main(whisper_size, input_dir, output_file):
    model = load_whisper_model(whisper_size)
    speaker_annos = []

    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    audio_files = os.listdir(input_dir)
    total_files = len(audio_files)
    processed_files = 0

    for audio_file in audio_files:
        audio_path = os.path.join(input_dir, audio_file)
        if os.path.isfile(audio_path):  
            wav, sr = process_audio_file(audio_path, target_sr)
            if wav is None:
                continue

            save_path = os.path.join(input_dir, f"processed_{processed_files}.wav")
            torchaudio.save(save_path, wav, target_sr, channels_first=True)
            lang, text = transcribe(model, save_path)
            if lang is None:
                continue
            text = "[EN]" + text + "[EN]\n"
            speaker_annos.append(f"{save_path}|{audio_file}|{text}")

            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")

    if len(speaker_annos) == 0:
        print("Warning: no English audios found.")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_size", default="medium")
    parser.add_argument("--input_dir", default="./denoised_audio/") 
    parser.add_argument("--output_file", default="labels.txt")
    args = parser.parse_args()

    main(args.whisper_size, args.input_dir, args.output_file)