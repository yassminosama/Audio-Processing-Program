from tkinter import *
from tkinter import filedialog
import pygame
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import customtkinter
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pyaudio
import struct
import scipy.signal as signal
from scipy.fftpack import fft


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()
root.title("Yaso Player")
root.iconbitmap("gui/yaso.ico")
root.geometry("1200*800")
root.resizable(True, True)

pygame.mixer.init()

global filename
filename = NONE

def add_song():
    global filename
    filenames = filedialog.askopenfilenames(initialdir="/", title="Choose Audio File", filetypes=[("Audio Files", "*.mp3 *.wav")])
    for filename in filenames:
        if filename.lower().endswith('.mp3'):
            # Convert MP3 to WAV format
            sound = AudioSegment.from_mp3(filename)
            new_filename = filename[:-4] + ".wav"  # Change file extension to WAV
            sound.export(new_filename, format="wav")
            filename = new_filename
            print("MP3 file converted to WAV format.")
        wav = wave.open(filename, "r")
        if wav.getnchannels() == 2:
            # Convert stereo file to mono
            sound = AudioSegment.from_wav(filename)
            sound = sound.set_channels(1)  # Convert to mono
            sound.export(filename, format="wav")
            print("Stereo file converted to mono.")
        song_box.insert(END, filename)
        print("File added: " + os.path.basename(filename).replace(".wav", ""))

def next_song():
    next_one = song_box.curselection()[0] + 1
    song_box.selection_clear(ACTIVE)
    song_box.activate(next_one)
    song_box.selection_set(next_one, last=None)
    play()

def previous_song():
    previous_one = song_box.curselection()[0] - 1
    song_box.selection_clear(ACTIVE)
    song_box.activate(previous_one)
    song_box.selection_set(previous_one, last=None)
    play()

def delete_song():
    global filename
    filename = NONE
    song_box.delete(ANCHOR)
    plot_combined(canvas, filename)
    stop()

def clear_playlist():
    global filename
    filename = NONE
    song_box.delete(0, END)
    plot_combined(canvas, filename)
    stop()


# Create a frame for the list box on the left side
left_frame = customtkinter.CTkFrame(root, width=30, height=90)
left_frame.pack(side="left")

song_box = Listbox(left_frame, width=30, height=90, selectmode=SINGLE, bg="white", fg="purple", selectbackground="lightblue", selectforeground="midnight blue")
song_box.pack(side="left", padx=10, pady=10)



def stop():
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)
    stop_recording(config)

def plot_combined(canvas, filename):
    # Clear existing canvas if any
    for widget in canvas.winfo_children():
        widget.destroy()

    if filename is None:
        return  # Don't plot anything if filename is None

    # Open the WAV file
    wav = wave.open(filename, "r")
    sample_rate = wav.getframerate()
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    # Time-domain data
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    # Frequency-domain data
    N = len(signal)
    yf = fft(signal)
    xf = np.linspace(0.0, sample_rate / 2.0, N // 2)

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Time-domain plot
    ax1.plot(time, signal, color="lightblue", linewidth=0.5)
    ax1.set_title("Time Domain")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time")

    # Frequency-domain plot
    ax2.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), color="red", linewidth=0.5)
    ax2.set_title("Frequency Domain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")

    # Embed the combined plot in the CustomTkinter canvas
    canvas_plot = FigureCanvasTkAgg(fig, master=canvas)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(side="top", fill="both", expand=True)


canvas_frame = customtkinter.CTkFrame(root, width=500, height=500)
canvas_frame.pack(side="top", fill="both", expand=True)
canvas = customtkinter.CTkFrame(canvas_frame, width=500, height=500)
canvas.pack(side="top", fill="both", expand=True)

def play():
    global filename
    filename = song_box.get(ACTIVE)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    plot_combined(canvas, filename)

global i
i = True
def pause():
    global i
    if i == True:
        pygame.mixer.music.pause()
        i = False
    else:
        pygame.mixer.music.unpause()
        i = True

def reverse_audio():
    audio = AudioSegment.from_file(filename)
    reversed_audio = audio.reverse()
    output_file = "reversed_audio_temp.wav"
    reversed_audio.export(output_file, format="wav")
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    plot_combined(canvas, filename)

class RecordingConfig:
    def __init__(self, chunk=1024, sample_format=pyaudio.paInt16, channels=1, fs=44100):
        self.chunk = chunk
        self.sample_format = sample_format
        self.channels = channels
        self.fs = fs
        self.frames = []
        self.isrecording = False
        self.filename = "rec.wav"

def start_recording(config):
    config.p = pyaudio.PyAudio()
    config.stream = config.p.open(
        format=config.sample_format,
        channels=config.channels,
        rate=config.fs,
        frames_per_buffer=config.chunk,
        input=True
    )
    config.isrecording = True
    print("Recording")
    recording_thread = threading.Thread(target=record_audio, args=(config,))
    plotting_thread = threading.Thread(target=plot_audio, args=(config,))
    recording_thread.start()
    plotting_thread.start()

def stop_recording(config):
    config.isrecording = False
    print("Recording complete")
    config.stream.stop_stream()
    config.stream.close()
    config.p.terminate()
    wf = wave.open(config.filename, "wb")
    wf.setnchannels(config.channels)
    wf.setsampwidth(config.p.get_sample_size(config.sample_format))
    wf.setframerate(config.fs)
    wf.writeframes(b"".join(config.frames))
    wf.close()
    song_box.insert(END, config.filename)

def record_audio(config):
    while config.isrecording:
        data = config.stream.read(config.chunk)
        config.frames.append(data)

def plot_audio(config):
    CHUNK = config.chunk
    RATE = config.fs
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    x = np.arange(0, 2 * CHUNK, 2)
    x_fft = np.linspace(0, RATE, CHUNK)
    line, = ax1.plot(x, np.random.rand(CHUNK), "r")
    line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK), "b")
    ax1.set_ylim(-20000, 20000)
    ax1.set_xlim(0, 2 * CHUNK)
    ax2.set_xlim(20, RATE / 2)
    ax2.set_ylim(0, 1)
    plt.ion()
    plt.show()
    while config.isrecording:
        data = config.stream.read(CHUNK)
        data_int = struct.unpack(f"{CHUNK}h", data)
        line.set_ydata(data_int)
        fft_data = np.abs(np.fft.fft(data_int)) * 2 / (11000 * CHUNK)
        line_fft.set_ydata(fft_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close(fig)

def txt():
    os.system('python txt_to_speach.py')

def talk():
    m=customtkinter.CTk()
    m.resizable(0,0)
    m.title('Web Search using voice')
    m.geometry('300x200')
    head=customtkinter.CTkLabel(m,text="search with your voice")
    head.pack(pady=20)
    def mic():
        import speech_recognition as sr
        import webbrowser
        with sr.Microphone() as source:
            sr.Microphone()
            r = sr.Recognizer()
            r.energy_threshold = 5000
            print("Speak!")
            audio = r.listen(source, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio)
                print("You said : {}".format(text))
                label6=customtkinter.CTkLabel(m,text="You said : {}".format(text))
                label6.pack()
                url = 'https://www.google.com/search?q='
                search_url = url+text

                #webbrowser.open() open your default web browser with a given url.
                webbrowser.open(search_url)
            except:
                label2=customtkinter.CTkLabel(m,text="Can't Recognize")
                label2.pack()
                print("Can't Recognize")
    btn1=customtkinter.CTkButton(m,text="Speak",command=mic,width=120)
    btn1.pack(pady=10)
    m.mainloop()

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed)})
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def speed():
    customtkinter.set_default_color_theme("blue")
    root=customtkinter.CTk()
    root.geometry("150x130")
    my_label=customtkinter.CTkLabel(root,text="Enter the speed")
    my_label.pack(pady=5,padx=5)
    rate= customtkinter.CTkEntry(root)
    def P():
        sound = AudioSegment.from_file(filename)
        fast_sound = speed_change(sound, float(rate.get()))
        output_file = "speed.wav"
        fast_sound.export(output_file, format="wav")
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        plot_combined(canvas, output_file)
        
    btn5=customtkinter.CTkButton(root,text="Done",command=P)
    rate.pack(pady=5,padx=5)
    btn5.pack(pady=5,padx=5)
    root.mainloop()

def apply_low_pass_filter():
    output_file = "low_pass_filtered_audio_temp.wav"
    sample_rate, data = wavfile.read(filename)
    nyquist = 0.5 * sample_rate
    cutoff = 3000
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    wavfile.write(output_file, sample_rate, filtered_data.astype(np.int16))
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    plot_combined(canvas, filename)

def apply_high_pass_filter():
    output_file = "high_pass_filtered_temp.wav"
    sample_rate, data = wavfile.read(filename)
    nyquist = 0.5 * sample_rate
    cutoff = 500
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    wavfile.write(output_file, sample_rate, filtered_data.astype(np.int16))

    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    plot_combined(canvas,output_file)

def apply_band_pass_filter():
    output_file = "band_pass_filtered_audio_temp.wav"
    sample_rate, data = wavfile.read(filename)
    nyquist = 0.5 * sample_rate
    low_cutoff = 500
    high_cutoff = 3000
    normal_cutoff = [low_cutoff / nyquist, high_cutoff / nyquist]
    b, a = signal.butter(5, normal_cutoff, btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    wavfile.write(output_file, sample_rate, filtered_data.astype(np.int16))
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    plot_combined(canvas, output_file)

def apply_notch_filter():
    def p():
        notch_freq_val = float(notch_freq.get())  # Get the input from the entry widget and convert to float
        output_file = "notch_filtered_audio_temp.wav"
        sample_rate, data = wavfile.read(filename)

        nyquist = 0.5 * sample_rate
        norm_notch_freq = notch_freq_val / nyquist

        # Quality factor
        Q = 30.0  # Higher Q factor means narrower notch

        # Design the notch filter
        b, a = signal.iirnotch(norm_notch_freq, Q)

        # Apply the notch filter
        filtered_data = signal.filtfilt(b, a, data)

        # Write the filtered data to a new file
        wavfile.write(output_file, sample_rate, filtered_data.astype(np.int16))

        # Play the filtered audio
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        plot_filtered_frequency_dbhz(sample_rate, data, filtered_data)
        
        root.destroy()  # Close the window after applying the filter

    root = customtkinter.CTk()
    root.geometry("300x150")
    root.title("Notch Filter")

    my_label = customtkinter.CTkLabel(root, text="Enter the frequency to be filtered out (in Hz):")
    my_label.pack(pady=10)

    notch_freq = customtkinter.CTkEntry(root)
    notch_freq.pack(pady=10)

    btn5 = customtkinter.CTkButton(root, text="Apply Filter", command=p)
    btn5.pack(pady=10)

    root.mainloop()

def plot_filtered_frequency_dbhz(sample_rate, original_data, filtered_data):
    N = len(original_data)
    
    # FFT of original data
    yf_original = fft(original_data)
    xf = np.linspace(0.0, sample_rate / 2.0, N // 2)
    
    # FFT of filtered data
    yf_filtered = fft(filtered_data)
    
    # Calculate dB/Hz
    dbhz_original = 20 * np.log10(np.abs(yf_original[:N // 2]) / np.sqrt(N))
    dbhz_filtered = 20 * np.log10(np.abs(yf_filtered[:N // 2]) / np.sqrt(N))
    
    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot original data in dB/Hz
    ax1.plot(xf, dbhz_original, color="blue", linewidth=0.5)
    ax1.set_title("Original Data in dB/Hz")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude (dB/Hz)")
    
    # Plot filtered data in dB/Hz
    ax2.plot(xf, dbhz_filtered, color="green", linewidth=0.5)
    ax2.set_title("Filtered Data in dB/Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (dB/Hz)")
    
    plt.tight_layout()
    plt.show()


btn_frame = customtkinter.CTkFrame(root, width=150, height=150)
btn_frame.pack(side="top", pady=20)

low_pass_filter_button = customtkinter.CTkButton(btn_frame, text="Low-Pass Filter",fg_color="purple" ,command=apply_low_pass_filter)
low_pass_filter_button.grid(row=1, column=0, padx=10, pady=10)

high_pass_filter_button = customtkinter.CTkButton(btn_frame, text="High-Pass Filter",fg_color="purple" ,command=apply_high_pass_filter)
high_pass_filter_button.grid(row=1, column=1, padx=10, pady=10)

band_pass_filter_button = customtkinter.CTkButton(btn_frame, text="Band-Pass Filter", fg_color="purple" ,command=apply_band_pass_filter)
band_pass_filter_button.grid(row=1, column=2, padx=10, pady=10)

notch_filter_button = customtkinter.CTkButton(btn_frame, text="Notch Filter",fg_color="purple" ,command=apply_notch_filter)
notch_filter_button.grid(row=1, column=3, padx=10, pady=10)
my_menu = Menu(root)
root.config(menu=my_menu)
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Options", menu=add_song_menu)
add_song_menu.add_command(label="Add one song to playlist", command=add_song)
add_song_menu.add_command(label="Google search", command=talk)
add_song_menu.add_command(label="text to speach", command=txt)

remove_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Remove Songs", menu=remove_song_menu)
remove_song_menu.add_command(label="Delete a song from playlist", command=delete_song)
remove_song_menu.add_command(label="Delete all songs from playlist", command=clear_playlist)

controls_frame = Frame(root)
controls_frame.pack()


play_btn_img = PhotoImage(file="gui/play.png").subsample(7, 7)
stop_btn_img = PhotoImage(file="gui/stop.png").subsample(7, 7)
next_btn_img = PhotoImage(file="gui/next.png").subsample(7, 7)
back_btn_img = PhotoImage(file="gui/back.png").subsample(7, 7)
pause_btn_img = PhotoImage(file="gui/pause.png").subsample(7, 7)
reverse_btn_img = PhotoImage(file="gui/reverse.png").subsample(7, 7)
record_btn_img = PhotoImage(file="gui/record.png").subsample(7, 7)
speed_btn_img = PhotoImage(file="gui/speed.png").subsample(7, 7)
back_button = Button(controls_frame, image=back_btn_img, borderwidth=0, command=previous_song)
forward_button = Button(controls_frame, image=next_btn_img, borderwidth=0, command=next_song)
play_btn = Button(controls_frame, image=play_btn_img, borderwidth=0, command=play)
pause_button = Button(controls_frame, image=pause_btn_img, borderwidth=0, command=pause)
stop_button = Button(controls_frame, image=stop_btn_img, borderwidth=0, command=stop)
record_button = Button(controls_frame, image=record_btn_img, borderwidth=0, command=lambda: start_recording(config))
speed_button = Button(controls_frame, image=speed_btn_img, borderwidth=0, command=speed)
reverse_button = Button(controls_frame, image=reverse_btn_img, borderwidth=0, command=reverse_audio)
back_button.grid(row=0, column=0, padx=10)
forward_button.grid(row=0, column=1, padx=10)
play_btn.grid(row=0, column=2, padx=10)
pause_button.grid(row=0, column=3, padx=10)
stop_button.grid(row=0, column=4, padx=10)
record_button.grid(row=0, column=5, padx=10)
reverse_button.grid(row=0, column=6, padx=10)
speed_button.grid(row=0, column=7, padx=10)

global config
config = RecordingConfig()

root.mainloop()