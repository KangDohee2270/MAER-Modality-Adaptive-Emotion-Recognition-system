# MAER: Modality Adaptive Emotion Recognition system
<p align='center'>
  <img src="https://user-images.githubusercontent.com/39416550/227125521-89933365-0255-432e-9803-137db362a2a0.gif" width="800"/>
</p>

- Inner emotion recognition framework through real-time fusion of audio, video, and biological signal
## Requirement
* Software
  - Window OS
  - Python(>=3.8)
  - tensorflow(>=2.8.0)
* Hardware
  - webcam
  - Shimmer3
  - mic

To install all dependencies, do this.
```
pip install -r requirements.txt
```

## Pretrained weights
- Pretrained weights for each modality can get from [here](https://drive.google.com/drive/u/0/folders/1mz9mqqq8DkHD-4z6dDit20mGg_nmAXYy)
- Create a "models" subfolder in each signal (i.e. audio, video and bio-signal) folder and save the downloaded model weights.
```bash
# Save pretrained weights
./
├── audio
│   ├── models
│   │   ├── arousal
│   │   |   ├──  [...]
│   │   ├── valence
│   │   |   ├──  [...]
├── video
│   ├── models
│   │   ├──  [...]
│   │   ├──  [...]
├── bio-signal
│   ├── models
│   │   ├── arousal
│   │   |   ├──  [...]
│   │   ├── valence
│   │   |   ├──  [...]
```
## Run
1. Run toy example.

  - Note that video, audio and bio sianals are already prepared in `example_record` folder.

  - The purpose of executing this toy example is two folds: 1) Verify each modality encoder's performance, and 2) Check the effectiveness of adaptive fusion (AF). 

    ```bash
    python example.py
    ```

2. Run real-time demo.

  - Details are as follows:

    (1) Attach the __Shimmer3__ to your left hand and connect it to your PC via Bluetooth.

    (2) Go to `/shimmer_data` and execute 'ShimmerMonitor.exe'

    (3) After connecting Shimmer3 and framework, run this code.

    ```bash
    python main.py
    ```

## Milestone
- [x] Code refactoring
- [x] Upload pre-trained weights
- [x] Initial update
