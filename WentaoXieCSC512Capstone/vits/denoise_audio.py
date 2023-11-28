import os
import json
import torchaudio

def separate_vocals(input_audio_path, output_audio_path, target_sr):
    # Separate vocals using demucs
    os.system(f"demucs --two-stems=vocals {input_audio_path}")

    # Load separated vocals and perform processing
    file = os.path.splitext(os.path.basename(input_audio_path))[0]
    vocals_path = f"./separated/htdemucs/{file}/vocals.wav"
    wav, sr = torchaudio.load(vocals_path, normalize=True, channels_first=True)

    # Merge two channels into one
    wav = wav.mean(dim=0).unsqueeze(0)

    # Resample if necessary
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    # Save denoised audio
    torchaudio.save(output_audio_path, wav, target_sr, channels_first=True)

def main(raw_audio_dir, denoise_audio_dir, config_path):
    # Get the target sampling rate from the config file
    with open(config_path, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    # List all files in the raw audio directory
    filelist = [file for file in os.listdir(raw_audio_dir) if file.endswith(".wav")]

    # Process each audio file
    for file in filelist:
        input_audio_path = os.path.join(raw_audio_dir, file)
        output_audio_path = os.path.join(denoise_audio_dir, os.path.splitext(file)[0] + "_denoised.wav")
        separate_vocals(input_audio_path, output_audio_path, target_sr)

if __name__ == "__main__":
    raw_audio_dir = "./raw_audio/"
    denoise_audio_dir = "./denoised_audio/"
    config_path = "./configs/finetune_speaker.json"
    main(raw_audio_dir, denoise_audio_dir, config_path)