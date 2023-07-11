from eeg_udp import EEGUDP

eeg = EEGUDP()

while True:
    print(eeg.reading())