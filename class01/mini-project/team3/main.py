## generate image ##
import using_gemini
import text2img


## generate tts ##
from initialize_tts import *
from pipeline_tts import *
import time
from bark import SAMPLE_RATE
import scipy.io.wavfile
import os

def make_tts():
    tokenizer, text_encoder_path0, text_encoder_path1, coarse_encoder_path, fine_model_dir = initialize()

    core = ov.Core()

    ov_text_model = OVBarkTextEncoder(
        core, "AUTO", text_encoder_path0, text_encoder_path1
    )
    ov_coarse_model = OVBarkEncoder(core, "AUTO", coarse_encoder_path)
    ov_fine_model = OVBarkFineEncoder(core, "AUTO", fine_model_dir)

    torch.manual_seed(42)
    t0 = time.time()
    
    text_path = '../text/diary.txt'

    with open(text_path, 'r') as file:
        content = file.read()

    audio_array = generate_audio(tokenizer, ov_text_model, ov_coarse_model, ov_fine_model, content)
    generation_duration_s = time.time() - t0
    audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

#	print(f"took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

    directory = "../tts"
    os.makedirs(directory, exist_ok=True)

    scipy.io.wavfile.write("../tts/snowy_night.wav", rate=SAMPLE_RATE, data=audio_array)


## generate music ##
import generate_music

def make_music():
    generate_music.generate()

def main():
    ## Using gemini ##
    gemini = using_gemini.Using_gemini()
    gemini.get_keyword() 
    
    ## generate image ##
    inf = text2img.Inference()
    text_enc, unet_model, vae_decoder, vae_encoder = inf.prepare_inference()
    inf.do_inference(text_enc, unet_model, vae_decoder, vae_encoder)

    ## generate tts ##
    make_tts()

    ## generate music ##
    make_music()

if __name__ == "__main__":
   main()
