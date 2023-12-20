python3 ttt.py &&
python3 text_to_speech_demo.py \
    --input PARK.txt \
    -o output.wav \
    --model_duration public/forward-tacotron/forward-tacotron-duration-prediction/FP32/forward-tacotron-duration-prediction.xml \
    --model_forward public/forward-tacotron/forward-tacotron-regression/FP32/forward-tacotron-regression.xml \
    --model_upsample public/wavernn/wavernn-upsampler/FP32/wavernn-upsampler.xml \
    --model_rnn public/wavernn/wavernn-rnn/FP32/wavernn-rnn.xml
