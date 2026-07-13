import os
import torch
import pandas as pd
import random
import soundfile as sf
import numpy as np
import urllib.request
import tarfile
import io

def prepare_tiny_libri3mix(output_dir="data/tiny_libri3mix/wav16k/max/train", num_mixtures=10):
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    print(f"Streaming LibriSpeech from {url}...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    
    utterances = []
    speakers_seen = set()
    
    # Create directories
    mix_dir = os.path.join(output_dir, "mix_clean")
    s1_dir = os.path.join(output_dir, "s1")
    s2_dir = os.path.join(output_dir, "s2")
    s3_dir = os.path.join(output_dir, "s3")
    
    for d in [mix_dir, s1_dir, s2_dir, s3_dir]:
        os.makedirs(d, exist_ok=True)
        
    spk_utterance_counts = {}
    print("Extracting utterances on the fly...")
    with tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if member.name.endswith(".flac"):
                parts = member.name.split('/')
                spk_id = parts[2]
                
                # Limit to 3 utterances per speaker to ensure diversity
                if spk_utterance_counts.get(spk_id, 0) >= 3:
                    continue
                    
                f = tar.extractfile(member)
                audio_bytes = f.read()
                
                # Decode with soundfile
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Check length (3 to 8 seconds)
                if 3.0 < len(audio_array) / sr < 8.0:
                    spk_utterance_counts[spk_id] = spk_utterance_counts.get(spk_id, 0) + 1
                    speakers_seen.add(spk_id)
                    utterances.append({
                        'speaker_id': spk_id,
                        'audio': audio_array,
                        'sr': sr
                    })
                    if len(utterances) >= num_mixtures * 3:
                        break

    print(f"Collected {len(utterances)} utterances from {len(speakers_seen)} speakers.")
    
    csv_rows = []
    
    print("Mixing audio...")
    for i in range(num_mixtures):
        random.shuffle(utterances)
        selected = []
        spks_selected = set()
        for u in utterances:
            if u['speaker_id'] not in spks_selected:
                spks_selected.add(u['speaker_id'])
                selected.append(u)
            if len(selected) == 3:
                break
                
        max_len = max([len(u['audio']) for u in selected])
        
        sources = []
        for u in selected:
            arr = u['audio']
            padded = np.pad(arr, (0, max_len - len(arr)), mode='constant')
            sources.append(padded)
            
        mix = sources[0] + sources[1] + sources[2]
        
        # Max-normalize
        max_amp = np.max(np.abs(mix))
        if max_amp > 0.9:
            gain = 0.9 / max_amp
            mix *= gain
            sources[0] *= gain
            sources[1] *= gain
            sources[2] *= gain
            
        mix_id = f"tiny_mix_{i:02d}"
        
        mix_path = os.path.join(mix_dir, f"{mix_id}.wav")
        s1_path = os.path.join(s1_dir, f"{mix_id}.wav")
        s2_path = os.path.join(s2_dir, f"{mix_id}.wav")
        s3_path = os.path.join(s3_dir, f"{mix_id}.wav")
        
        sf.write(mix_path, mix, 16000, subtype='FLOAT')
        sf.write(s1_path, sources[0], 16000, subtype='FLOAT')
        sf.write(s2_path, sources[1], 16000, subtype='FLOAT')
        sf.write(s3_path, sources[2], 16000, subtype='FLOAT')
        
        csv_rows.append({
            "ID": mix_id,
            "duration": max_len / 16000.0,
            "mix_wav": os.path.abspath(mix_path),
            "mix_wav_format": "wav",
            "mix_wav_opts": "",
            "s1_wav": os.path.abspath(s1_path),
            "s1_wav_format": "wav",
            "s1_wav_opts": "",
            "s2_wav": os.path.abspath(s2_path),
            "s2_wav_format": "wav",
            "s2_wav_opts": "",
            "s3_wav": os.path.abspath(s3_path),
            "s3_wav_format": "wav",
            "s3_wav_opts": ""
        })
        
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "tiny_train.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Generated {num_mixtures} mixtures and saved CSV to {csv_path}")

if __name__ == "__main__":
    prepare_tiny_libri3mix()
