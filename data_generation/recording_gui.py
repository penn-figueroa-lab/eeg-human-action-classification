import time
import datetime
import tkinter as tk
from threading import Thread

from logger_custom import Logger
from eeg_udp import EEGUDP
from optitrack_ros import OPTITRACKROS
from webcam_ros import WEBCAMROS

root = tk.Tk()
w0, h0 = 1920, 1080  # monitor 1
w1, h1 = 1920, 1080  # monitor 2

file_name = ''
recording_status = False
cue_states = ['Rest', 'Ready', '←', '→', '↔', 'Left', 'Right', 'Both', 'Left (AUD)', 'Right (AUD)', 'Both (AUD)',
              'Left (TCH)', 'Right (TCH)', 'Both (TCH)']
cue_status = 0
cue_timing = 0
optitrack = OPTITRACKROS()
neurone = EEGUDP()
webcam = WEBCAMROS()


def show_win1():
    global recording_status
    global cue_states
    global cue_status
    if recording_status:
        if cue_status > 1 and cue_status < 5:
            font = ('Helvetica', 250, 'bold')
        else:
            font = ('Helvetica', 100, 'bold')
        display_win1.config(text=cue_states[cue_status], fg='Black')
    else:
        display_win1.config(text='Not recording', fg='Black', font=('Helvetica', 100, 'bold'))
    win1.after(10, show_win1)


def show_win2():
    global recording_status
    global cue_states
    global cue_status
    global file_name
    global cue_timing
    if recording_status:
        display_win0.config(text=file_name + '\n' + cue_states[cue_status] + '\n' + str(round(cue_timing, 2)), fg='Black')
    else:
        display_win0.config(text=file_name + '\n' + 'Not recording', fg='Black')
    win0.after(10, show_win2)


def recording_thread():
    global recording_status
    if not recording_status:
        recording_status = True
        thread_recording = Thread(target=recording)
        thread_recording.start()


def recording():
    global file_name
    global recording_status
    global cue_states
    global cue_status
    global cue_timing

    file_name = file_name_win0.get(1.0, "end-1c")

    global optitrack
    global neurone

    import random
    random.seed(10)
    
    # resting (without shield)
    if file_name == '001':
        cues = [[0, 60]]
    # resting (with shield)
    elif file_name == '002':
        cues = [[0, 60]]
    # eye moving (with shield)
    elif file_name == '003':
        cues = [[0, 60]]
    # head moving (with shield)
    elif file_name == '004':
        cues = [[0, 60]]
    # grasping (reaching)
    elif file_name == '101':
        random_sequence = [5, 6, 7] * 10
        random.shuffle(random_sequence)
        cues = [[0, 10]]
        for i in random_sequence:
            cues += [[1, 2], [i, 4]]
    # grasping (fisting)
    elif file_name == '102':
        random_sequence = [5, 6, 7] * 10
        random.shuffle(random_sequence)
        cues = [[0, 10]]
        for i in random_sequence:
            cues += [[1, 2], [i, 2]]
    # without cues (reaching + fisting)
    elif file_name == '103':
        cues = [[1, 120]]
    elif file_name == '000':
        random_sequence = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        cues = [[0, 2]]
        for i in random_sequence:
            cues += [[i, 2]]
    else:
        cues = []

    logger = Logger()
    logger.start(file_name)

    for i, (cue, interval) in enumerate(cues):
        cue_status = cue
        time_list = datetime.datetime.now()
        time_list = [time_list.year, time_list.day, time_list.hour,
                     time_list.minute, time_list.second, time_list.microsecond]
        logger.cue_write({'cue': cue_states[cue_status], 'datatime': time_list})
        time_start = time.time()
        cue_timing = interval
        while time.time() - time_start < interval:#cue_timing < 0:
            eeg_reading = neurone.reading()
            optitrack_reading = optitrack.reading()
            webcam_reading = webcam.reading()

            if optitrack_reading:
                time_list = datetime.datetime.now()
                time_list = [time_list.year, time_list.day, time_list.hour,
                             time_list.minute, time_list.second, time_list.microsecond]
                data_dict = {**eeg_reading, **optitrack_reading, **webcam_reading, **{'cue': cue, 'cue_index': i}, **{'datetime': time_list}}
                # data_dict = {**eeg_reading, **{'datetime': time_list}, **{'cue': cue, 'cue_index': i}}
                logger.write(data_dict)
                optitrack.rate.sleep()

            cue_timing = interval - (time.time() - time_start)

    logger.close()
    recording_status = False


def close(event):
    root.quit()


root.wm_overrideredirect(True)

win0 = tk.Toplevel()
win0.geometry("400x200")
file_name_win0 = tk.Text(win0, height=1, width=20)
file_name_win0.pack()
start_button_win0 = tk.Button(win0, text="Recording start", command=recording_thread)
start_button_win0.pack()
display_win0 = tk.Label(win0, text='')
display_win0.pack()

win1 = tk.Toplevel()
win1.geometry(f"{w1}x{h1}+{w0}+0")
win1.attributes("-fullscreen", True)
display_win1 = tk.Label(win1, text='', font=('Helvetica', 100, 'bold'))
display_win1.pack(expand=True)
win1.bind("<Button-1>", close)

show_win1()
show_win2()
root.mainloop()
