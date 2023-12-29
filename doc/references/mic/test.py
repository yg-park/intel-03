#!/usr/bin/python3


# 오디오 시스템과 상호작용하기 위해 PyAudio 모듈을 임포트합니다
import pyaudio
# WAV 파일을 다루기 위해 wave 모듈을 임포트합니다
import wave


"""
# 오디오 작업을 관리하기 위한 PyAudio의 인스턴스를 생성합니다
p = pyaudio.PyAudio()

# 호스트의 오디오 시스템 정보를 검색합니다 (예: 사운드 카드)
info = p.get_host_api_info_by_index(0)
# 사용 가능한 오디오 장치의 수를 얻습니다
num_devices = info.get('deviceCount')

# 각 오디오 장치를 반복하여
for i in range(0, num_devices):
    # 현재 장치가 입력을 지원하는지 확인합니다 (즉, 오디오를 녹음할 수 있는지)
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        # 입력 가능한 장치의 ID와 이름을 출력합니다
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# PyAudio 인스턴스를 안전하게 종료합니다
p.terminate()
"""


# 녹음에 사용될 장치의 ID를 설정합니다. Depending on your system
# You might use the index of "USB2.0 PC CAMERA: Audio (hw:2,0)"
DEVICE_ID = 0

# 녹음의 형식을 설정합니다 (이 경우 16비트 PCM)
FORMAT = pyaudio.paInt16
# 채널 수를 설정합니다 (모노의 경우 1)
CHANNELS = 1
# 샘플 레이트를 설정합니다 (Hz 단위)
RATE = 48000
# 버퍼당 프레임 수를 설정합니다
CHUNK = 1024
# 녹음 시간을 초 단위로 설정합니다
RECORD_SECONDS = 5
# 출력 WAV 파일의 이름입니다
WAVE_OUTPUT_FILENAME = "output.wav"

# PyAudio 인스턴스를 다시 생성합니다
audio = pyaudio.PyAudio()

# 녹음을 위한 스트림을 엽니다
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=DEVICE_ID)
print("recording...")
frames = []

# 녹음된 오디오를 청크로 녹음하여 리스트에 저장합니다
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# 스트림을 닫고 PyAudio 인스턴스를 종료합니다
stream.stop_stream()
stream.close()
audio.terminate()

# 쓰기 모드로 출력 WAV 파일을 엽니다
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# WAV 파일의 채널 수를 설정합니다
waveFile.setnchannels(CHANNELS)
# 샘플 폭을 설정합니다 (바이트 단위)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# 프레임 레이트를 설정합니다 (Hz 단위)
waveFile.setframerate(RATE)
# 녹음된 프레임을 파일에 씁니다
waveFile.writeframes(b''.join(frames))
# WAV 파일을 닫습니다
waveFile.close()

