import pyaudio
import wave
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from json_tricks import dump, load
from tensorflow.keras.models import Sequential, Model, model_from_json


def remove_silence(x, fs, thresh):
    frame_size = round(0.1*fs)
    N = len(x)
    nframes = round(N/frame_size)
    x_sil = []
    inx1= 1
    inx2 = 0
    for i in range(nframes):
        frame = x[((inx1-1)*frame_size+1) : (frame_size*inx1)]
        inx1 = inx1 + 1
        maximum = np.max(frame)

        if maximum > thresh:
            inx = inx2+1
            x_sil[(inx2-1)*frame_size+1 : frame_size*inx2] = frame

    x_sil = np.array(x_sil)

    return x_sil


# -------------------------------------------------------
# PARAMS


channels = 2
filename = "output.wav"
# -------------------------------------------------------


def pre_process(file, fs=16000, s=3):
    L = fs*s
    fade_len = round(int(fs / 64))
    fade_leno=round(int(fs / 32))
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_leno)
    x, fs = librosa.load('output.wav', res_type='kaiser_fast', duration=s, sr=fs)
   
    x[0:fade_len] = x[0:fade_len] * fade_in
    x[x.size-fade_leno:x.size] = x[x.size-fade_leno:x.size] * fade_out
    x = np.expand_dims(x, axis=1)
    norm_x = sklearn.preprocessing.normalize(x, norm='max', axis=0)
    trim_X = remove_silence(norm_x, fs, 0.01)
    trim_x = trim_X.flatten()

    if L > len(x - 1):
        pad_x = np.pad(trim_x, (0, L - len(trim_x)), 'constant')
    else:
        pad_x = trim_x[0:L]

    fin_x = nr.reduce_noise(pad_x, sr = 16000)
    mfccs = librosa.feature.mfcc(y=fin_x, sr=fs, n_mfcc=13, hop_length=256)
    mels = librosa.feature.melspectrogram(y=fin_x, sr=fs, n_fft=2048, hop_length=256)
    features = np.asarray(mels).astype('float32')
    features = np.expand_dims(features, axis=0)
    features = np.swapaxes(features,1, 2)
    np.nan_to_num(features, nan=0)

    return features




def plot(features):
    fig, ax = plt.subplots() 
    S_dB = librosa.power_to_db(features, ref=np.max) 
    img = librosa.display.specshow(S_dB, x_axis='time', 
                            y_axis='mel', sr=16000, 
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()



def record_save(length=3, fs=16000, chunksize=256, format=pyaudio.paInt16 ):
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print("Recording Now...")

    stream = p.open(format=format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunksize,
                input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunksize * length)):
        data = stream.read(chunksize)
        frames.append(data)


    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()