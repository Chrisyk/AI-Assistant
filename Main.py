import os
from openai import OpenAI
from chronological import read_prompt
import whisper
import queue
import threading
from playsound import playsound
from gtts import gTTS
import torch
from recordingFunc import record_audio, transcribe_forever

characterName = "Assistant"
micLanguage = 'english'
language = 'en'

if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

def getMic():
    audio_model = whisper.load_model("base", device)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    threading.Thread(target=record_audio,
                     args=(audio_queue, 300, 0.8, False)).start()
    threading.Thread(target=transcribe_forever,
                     args=(audio_queue, result_queue, audio_model, micLanguage, False)).start()

    return result_queue.get()


client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

question = []
i = 0

# Assistant

assistant = client.beta.assistants.create(
  name=characterName,
  instructions= read_prompt("Default"),
  model="gpt-4-1106-preview",
)

thread = client.beta.threads.create()

while True:
    
    question.insert(i, getMic())

    print("Me:" + question[i])

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question[i]
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
        
    )

    print ("Running")

    while (run.status != "completed"):
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    response = messages.data[0].content[0].text.value

    print (characterName + ": " + response)

    myobj = gTTS(text = response, lang=language, slow=False)
    myobj.save("response.mp3")
    playsound("response.mp3")
    os.remove("response.mp3")
    
    i += 1
    

