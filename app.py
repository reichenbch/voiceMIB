import time
import torch
import requests
import numpy as np
import gradio as gr
from pydub import AudioSegment
from transformers import pipeline
from usellm import Message, Options, UseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperTokenizerFast, WhisperForConditionalGeneration

tokenizer1 = AutoTokenizer.from_pretrained("microsoft/biogpt", add_special_tokens=False, fast=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to('cuda:0')


def text_to_speech(text_input):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "dec1209f94bd83c3f4135dac5358a3c2"
    }
    
    data = {
        "text": text_input,
        "model_id": "eleven_monolingual_v1"
    }
    
    audio_write_path = f"""output_{int(time.time())}.mp3"""
    
    response = requests.post(url, json=data, headers=headers)
    with open(audio_write_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    return audio_write_path


def whisper_inference(input_audio):
    
    pipe = pipeline(task="automatic-speech-recognition",model='openai/whisper-large', device='cuda:0')
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language='en', task="transcribe")
    transcription = pipe(input_audio)['text']

    return transcription



def biogpt_large_infer(input_text):
    
    inputs = tokenizer1(input_text, return_tensors="pt").to('cuda:0')

    with torch.no_grad():
        beam_output = model.generate(**inputs,
                                    min_length=100,
                                     max_length=1024,
                                     num_beams=5,
                                     early_stopping=True
                                    ).to('cuda:0')
    output = tokenizer1.decode(beam_output[0], skip_special_tokens=True)
    torch.cuda.empty_cache()

    output = output.replace('▃','').replace('FREETEXT','').replace('TITLE','').replace('PARAGRAPH','').replace('ABSTRACT','').replace('<','').replace('>','').replace('/','').strip()
    
    return output


    
def chatgpt_infer(input_text):

    service = UseLLM(service_url="https://usellm.org/api/llm")

    messages = [
      Message(role="system", content="You are a medical assistant, which answers the query based on factual medical information only."),
      Message(role="user", content=f"Give me few points on the disease {input_text} and its treatment."),
    ]
    
    options = Options(messages=messages)

    response = service.chat(options)
    
    return response.content


def audio_interface_demo(input_audio):
    
    en_prompt = whisper_inference(input_audio)
    
    biogpt_output = biogpt_large_infer(en_prompt)
    chatgpt_output = chatgpt_infer(en_prompt)
    
    bio_audio_output = text_to_speech(biogpt_output)
    chat_audio_output = text_to_speech(chatgpt_output)
            
    return biogpt_output, str(bio_audio_output), chatgpt_output, str(chat_audio_output)


def text_interface_demo(input_text):
    
    biogpt_output = biogpt_large_infer(input_text)
    chatgpt_output = chatgpt_infer(input_text)
    
    return biogpt_output, chatgpt_output


examples = [
    ["Meningitis is"],
    ["Brain Tumour is"]  
]

app = gr.Blocks()
with app:
    gr.Markdown("# **<h4 align='center'>Voice based Medical Informational Bot<h4>**")
     
    with gr.Row():
        with gr.Column():
            
            with gr.Tab("Text"):
                input_text = gr.Textbox(lines=3, value="Brain Tumour is", label="Text")
                text_button = gr.Button(value="Generate")

            with gr.Tab("Audio"):
                input_audio = gr.Audio(source="microphone", type="filepath", label='Audio') #value="input.mp3",
                audio_button = gr.Button(value="Generate")  
                  
    with gr.Row():
        with gr.Column():
            with gr.Tab("Output Text"):

                biogpt_output1 = gr.Textbox(lines=3, label="BioGPT Output")
                chatgpt_output1 = gr.Textbox(lines=3,label="ChatGPT Output")

            with gr.Tab("Output Audio"):

                biogpt_output = gr.Textbox(lines=3, label="BioGPT Output")
                biogpt_audio_output = gr.Audio(value=None, label="BioGPT Audio Output")
                
                chatgpt_output = gr.Textbox(lines=3,label="ChatGPT Output")
                chatgpt_audio_output = gr.Audio(value=None, label="ChatGPT Audio Output")
                

    text_button.click(text_interface_demo, inputs=[input_text], outputs=[biogpt_output1, chatgpt_output1])
    audio_button.click(audio_interface_demo, inputs=[input_audio], outputs=[biogpt_output, biogpt_audio_output, chatgpt_output, chatgpt_audio_output])
    
app.launch(debug=True, share=True)