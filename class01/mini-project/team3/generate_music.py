import warnings

import openvino as ov
import numpy as np
import torch
from torch.jit import TracerWarning
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

# Ignore tracing warnings
warnings.filterwarnings("ignore", category=TracerWarning)

def generate():
    # Load the pipeline
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torchscript=True, return_dict=False)


    device = "cpu"
    sample_length = 8  # seconds

    n_tokens = sample_length * model.config.audio_encoder.frame_rate + 3
    sampling_rate = model.config.audio_encoder.sampling_rate
    #print('Sampling rate is', sampling_rate, 'Hz')

    model.to(device)
    model.eval();

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")


    # 파일 경로 지정
    file_path = '../text/keyword.txt'

    # 파일 열기 및 내용 읽기
    with open(file_path, 'r') as file:
        file_content = file.read()

    # "Musical Atmosphere:" 이후, "Keywords:" 이전의 내용 추출
    musical_atmosphere_start_index = file_content.find('Musical Atmosphere:')
    keywords_start_index = file_content.find('Keywords:', musical_atmosphere_start_index)

    if musical_atmosphere_start_index != -1 and keywords_start_index != -1:
        musical_atmosphere_section = file_content[musical_atmosphere_start_index:keywords_start_index].strip()


        inputs = processor(
            text = musical_atmosphere_section,
            return_tensors="pt",
        )

        #print(musical_atmosphere_section)

        audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=n_tokens)

        waveform = audio_values[0].cpu().squeeze() * 2**15
        waveform = waveform.numpy().astype(np.int16)
        scipy.io.wavfile.write("../music/music.wav", rate=sampling_rate, data=waveform)
