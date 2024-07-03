# eeg-human-action-classification
This repository contains example code for classifying human actions (specifically, left or right hand movements) using real EEG data. Additionally, it includes tools for integrating EEG data with the Robot Operating System (ROS).

## Getting Started
### Prerequisites
- Python 3.X
- [ROS 1 Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) (optional for ROS tools)

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/eeg-human-action-classification.git
cd eeg-human-action-classification
```
2. Install the required Python libraries
```bash
pip install -r requirements.txt
```

### Usage

1. EEG action classification:

Download the datasets under `data` folder from the [link](). Run the script to process EEG data and classify hand movements:
```bash
python ./scripts/train.py
```

2. EEG UDP ROS node for [Bittium NeurOne](https://www.bittium.com/medical/bittium-neurone)
To run the EEG UDP ROS node for Bittium NeurOne, use the following command:
```bash
python ./ros_utils/nerone_udp_node.py
```
You need to set the following arguments based on your device settings:
- `n_electrodes`: The number of electrodes to read.
- `packet_size`: The number of data points per electrode.
- `device`: The Bittium NeurOne device ("exg" or "tesla").
- `channel_type`: The type of channel reading electric type ("ac" or "dc").
- `udp_ip`: The IP address of the device (this must match the device software and the receiving computer's network settings).
- `udp_port`: The port number of the device (this must match the device software and the receiving computer's network settings).

- Example command:
```bash
python ./ros_utils/nerone_udp_node.py --n_electrodes 32 --packet_size 1 --device "tesla" --channel_type "ac" --udp_ip "192.168.200.240" --udp_port 50000
```

## License
This project is licensed under the Apache License Version 2.0. See the LICENSE file for details.
## Acknowledgement
These tools were originally developed for the following work.
```
@article{Choi2024OnTF,
  title={On the Feasibility of EEG-based Motor Intention Detection for Real-Time Robot Assistive Control},
  author={Ho Jin Choi and Satyajeet Das and Shaoting Peng and Ruzena Bajcsy and Nadia Figueroa},
  journal={ArXiv},
  year={2024},
  volume={abs/2403.08149},
  url={https://api.semanticscholar.org/CorpusID:268379125}
}
```