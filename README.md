# Neurocontrol

Neurocontrol is a real-time Brain-Computer Interface (BCI) system designed for seamless EEG signal acquisition, processing, classification, and control of external devices. It integrates microcontrollers, signal amplifiers, wireless communication modules, and machine learning to create a functional and scalable neuro-control environment.

---

## Features

  - Real-time EEG signal acquisition from scalp electrodes
  - Signal amplification and filtering using Bio Amp hardware
  - Wireless data transmission over UDP (ESP8266/ESP32)
  - User-specific model training and dynamic model loading
  - Interactive GUI (built with PySide6) for system control and EEG visualization
  - Relay-based control of external appliances via ESP
  - Modular hardware and software architecture

---

## System Architecture

  1. Signal Acquisition:
  
   - EEG signals are collected by electrodes and amplified by the Bio Amp Pill
   - Data is transmitted to the Maker Uno for digitization
   - Serially sent to the Raspberry Pi
  
  2. Signal Processing & Classification:
  
   - Bandpass and notch filtering to remove noise
   - Feature extraction and classification using scikit-learn
   - Models are stored per-user and dynamically loaded for predictions
     
  3. Wireless Communication:
  
   - UDP protocol used to transmit predictions to ESP8266
   - ESP8266 triggers relay module to control connected devices
  
  4. Graphical Interface:
  
   - Built with PySide6
   - Displays live EEG predictions and user management options

---
## Project Structure
```
Neurocontrol/
│
├── Maker UNO Signal Code/
│   └── Signal.INO                       # Arduino code for EEG signal acquisition
│
├── ESP8266 Code/
│   ├── otabasecode/
│   │   └── otabasecode.INO             # OTA base firmware
│   └── otaupdatecode/
│       └── otaupdatecode.INO           # OTA update firmware
│
├── users/
│   ├── user1/
│   │   ├── signal.csv                  # Raw EEG signal data
│   │   ├── model.pkl                   # Trained classification model
│   │   └── scalar.pkl                  # Scaler used during training
│   └── user2/
│       └── ...
│
├── electrode position setup.txt        # Notes on electrode placement
├── esp8266.json                        # Configuration file for ESP
├── main.py                             # Main GUI + ML application
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
└── LICENSE                             # Project license
```
---

##  Hardware Requirements

  - Raspberry Pi (with Touch Display)
  - Maker Uno (Arduino-compatible)
  - ESP8266/ESP32 module
  - Electrodes (3-point EEG system)
  - Bio Amp Pill (EEG signal amplifier)
  - Relay Module
  - 3.7V Li-ion Battery Packs

---

## Software Stack

 - Python 3.10+
   - scikit-learn, numpy, scipy, pandas, PySide6, pyserial, socket
 - C/C++
   - For Maker Uno and ESP programming (via Arduino IDE)
 - Ubuntu-based OS (Raspberry Pi OS)(64-bit)
   - Recommended for full compatibility with hardware and GUI

---

## Setup Instructions

  ### Hardware Setup
  
  - Place electrodes based on instructions in text file
  - Connect electrodes to Bio Amp and Bio Amp to Maker Uno
  - Connect Maker Uno to Raspberry Pi with serial communication
  - Connect Esp8266/32 with Relay
  - Connect Relay with external LED or light or any electronic that is limited to on and off function.
   
  ### Software Setup
  
  - on Maker Uno:
    - Upload Signal.INO to Maker Uno
  - on Esp8266/32:
    - To setup wifi, Upload otabasecode.INO and otaupdatecode.INO to ESP8266 
  - On Raspberry Pi:
    - Install python if already not available
    - Create python virtual environment
    - Install dependencies inside virtual environment
    - Run GUI: python main.py
  - Select already available user or create new
  - Begin live signal classification and control

---

## Model Management

  - EEG data and trained models are saved per-user inside the users/ folder
  - Use GUI to select active model for live prediction
  - All models are in .pkl format

---

## Communication Flow

  - Serial: Maker Uno → Raspberry Pi
  - UDP: Raspberry Pi → ESP8266
  - Relay: ESP8266 triggers relay module for device control

---

## License

This project is licensed under the MIT License.
