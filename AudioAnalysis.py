import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NOTE_MIN = 40        # E2: 吉他之最低音
NOTE_MAX = 69        # A4: 吉他取樣之最高音
FSAMP = 44100        # 取樣率(根據裝置輸入)
FRAME_SIZE = 4096    # 每個frame有4096個樣本數(特定時間段內捕獲的連續樣本數據，時間取決於取樣率)
FRAMES_PER_FFT = 8   # 取16個連續的frame來做
THRESHOLD = 100      # 振幅臨界值，超過了才會取(過濾雜訊)

# 每個FFT中的樣本數 4096*8 = 32768
SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT

# 相鄰頻率之間的距離，決定了在頻率域上能夠分辨的最小頻率變化 44100/32768 = 1.345
FREQ_STEP = float(FSAMP) / SAMPLES_PER_FFT  

# 設置音名
NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

def freq_to_note_number(frequency):
    return 69 + 12 * np.log2(frequency / 440.0)

def note_number_to_freq(note):
    #A4=440Hz, 每差n個半音頻率就差2^n倍
    return 440 * 2.0**((note - 69) / 12.0)

def note_name(note):  # 把音的值轉為音名
    return NOTE_NAMES[note % 12] + str(note // 12 - 1)

def note_to_fftbin(note):
    return note_number_to_freq(note) / FREQ_STEP

# 限制最高及最低音之音頻，降低不必要之計算
NOTE_MIN_BIN = max(0, int(np.floor(note_to_fftbin(NOTE_MIN - 1))))
NOTE_MAX_BIN = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX + 1))))

# 初始化一個FFT需要的空間
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)   #buf(32768,0)
num_frames = 0

# 設置麥克風輸入
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)

stream.start_stream()

# 畫圖
fig, ax = plt.subplots()
x = np.arange(0, 2 * FRAME_SIZE, 2)
line, = ax.plot(x, np.random.rand(FRAME_SIZE), '-', lw=2)
ax.set_ylim(-500, 500)  #振幅
ax.set_xlim(0, FRAME_SIZE)
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# 更新波形圖
def update_plot(frame):
    global buf, num_frames

    # 平移資料，舊資料往後推，新資料放前面
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
    buf[-FRAME_SIZE:] = np.frombuffer(stream.read(FRAME_SIZE), np.int16)

    # 計算FFT
    fft = np.fft.rfft(buf)

    # 取影響力最大之頻率
    max_response_index = np.abs(fft[NOTE_MIN_BIN:NOTE_MAX_BIN]).argmax()
    freq = (max_response_index + NOTE_MIN_BIN) * FREQ_STEP

    # 預估音高
    note_number = freq_to_note_number(freq)
    nearest_note = int(round(note_number))

    # 處理下一幀
    num_frames += 1

    if num_frames >= FRAMES_PER_FFT:
        if np.max(np.abs(buf)) > THRESHOLD: #過濾雜訊，以免印出過多不必要的資訊
            print('Frequency: {:7.2f} Hz     Note: {:>3s} {:+.2f}'.format(freq, note_name(nearest_note), note_number - nearest_note))

    # 更新新波形
    line.set_ydata(buf[-FRAME_SIZE:])
    return line,

ani = FuncAnimation(fig, update_plot, blit=True, interval=50)

plt.show()

stream.stop_stream()
stream.close()
pyaudio.PyAudio().terminate()