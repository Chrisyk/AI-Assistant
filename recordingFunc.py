import io
from pydub import AudioSegment
import speech_recognition as sr
import os
import torch
import numpy as np


def record_audio(audio_queue, energy, pause, dynamic_energy):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        
        #get and save audio to wav file
        audio = r.listen(source)

        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
        audio_data = torch_audio

        audio_queue.put_nowait(audio_data)


def transcribe_forever(audio_queue, result_queue, audio_model, lang, verbose):
    audio_data = audio_queue.get()
    result = audio_model.transcribe(audio_data,language=lang, fp16=False)

    if not verbose:
        predicted_text = result["text"]
        result_queue.put_nowait(predicted_text)
    else:
        result_queue.put_nowait(result)
