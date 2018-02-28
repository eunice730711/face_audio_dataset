import wave
import numpy as np
def read_wav_data(filename):
    '''
        读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    wave_data = wave_data 
    return num_channel, num_sample_width, framerate, num_frame, wave_data 

def mu_law_encoding(x, quantization_channels=256):
    mu = quantization_channels - 1.
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((x_mu + 1) / 2 * mu + 0.5).astype(int)

if(__name__=='__main__'):

    nchannels, samplewidth, fs, nframes, wave_data = read_wav_data('./output/aaa.wav')
    wave_data = wave_data/32768
    print(nframes)
    print(len(wave_data[1]))
    print(wave_data[0][:10])



