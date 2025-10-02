import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0:
        print(f"ID: {i}, Name: {info.get('name')}, Max Sample Rate: {info.get('defaultSampleRate')}")
p.terminate()
