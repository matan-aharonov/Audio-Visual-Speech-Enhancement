# Audio-Visual-Speech-Enhancement
Engineering Project

### Real-Time
A folder contains real time speech enhancer files.
* real_time_speech_enhancer -	The main file that runs the whole system in real time. In this file all work is done in parallel.
* real_time_audio_io.py and real_time_video_io.py - writing and manipulating audio and video files in real-time.
* real_time_network - neural network architecture with real-time code adjustments.
* real_time_face_detection.py - detecting face and mouth regions in images using dlib library in real time.

### Pre-Processing and Neural Network Files (Offline Mode)
* data_processor.py - contain methods that are responsible for the audio and video pre-processing.
* dataset.py - contains two classes that are responsible for processing audio and video data files from the computer in offline mode.
* network.py - the neural network architecture.
* speech_enhancer.py - The main file responsible for running the system in offline mode.

### Verification and Evaluation Files
* speech_enhancement_evaluator.py (PESQ)
* snr.py (SNR)
* pesq - executable program.

### mediaio
A python wrapper for reading, writing and manipulating audio and video files.

##### Dependencies
* python >= 2.7
* scipy >= 0.19.0
* numpy >= 1.12.1
* imageio >= 2.1.2
* opencv >= 3.2.0

### face-detection
A python wrapper for detecting face and mouth regions in images using dlib library.

##### Dependencies
* python >= 2.7
* numpy >= 1.12.1
* dlib >= 19.4.0
* opencv >= 3.2.0

### scripts
Scripts we use in data pre-processing and through the CS clusters with Python virtual environment.

### Year3
Last year's project tasks (hands on practice and training with python libraries and development tools).
