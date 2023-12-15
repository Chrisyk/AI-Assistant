import io
from pydub import AudioSegment
import speech_recognition as sr
import os
import torch
import numpy as np


def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        
        #get and save audio to wav file
        audio = r.listen(source)
        if save_file:
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            filename = os.path.join(temp_dir, f"temp{i}.wav")
            audio_clip.export(filename, format="wav")
            audio_data = filename
        else:
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio

        audio_queue.put_nowait(audio_data)


def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
    audio_data = audio_queue.get()
    if english:
        result = audio_model.transcribe(audio_data,language='english')
    else:
        result = audio_model.transcribe(audio_data)

    if not verbose:
        predicted_text = result["text"]
        result_queue.put_nowait(predicted_text)
    else:
        result_queue.put_nowait(result)

    if save_file:
        os.remove(audio_data)
