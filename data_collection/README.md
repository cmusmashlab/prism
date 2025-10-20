# 0. Setup
- Apple Watch (we use Series 7 or later)
- iPhone

## 0.1. How to use the watch app

1. Input `Participant ID` and `Session ID` on the phone app (anything is ok)    
2. Press `Pair watch` on the phone
    1. The watch needs to be well connected with the phone here
    2. You may need to tap the screen of the watch to *activate* it
3. You can press `Start` either on the phone or watch
4. You can press `Stop` either on the phone or watch
    1. At this point, the motion data is already saved on the phone
5. You can press `Audio Share` button on the watch to transfer audio file
    1. This may take time (e.g., 3 minutes for 10-minute recording)
    2. Both the watch and phone apps should be awake during the transfer
6. You can see the complete message on the phone when the audio file is successfully transferred
7. On `Files` app on the phone, you can access files.

- You can always reset both Watch and Phone apps to come back to the initial state
- Audio data transfer may not succeed sometimes due to many reasons (e.g., network connection). Don’t worry. The data is at least on the watch and they will be transferred in future attempts.

## 0.2. Define a procedural task

We need to define a task for the data collection. We do it carefully by considering the complexity of the task and expected sensing feasibility. The task consists of several steps, and we will create `steps.txt` to list each step.
We do not have to care about the order of the steps at this stage. 

Here is an example of `steps.txt` for the latte-making task.

```
attach filter
adjust grind
grind beans
move filter
place cup
brew coffee
use fridge
pour milk (jug)
steam milk
purge wand
remove cup
pour milk (cup)
wash jug
get towel
clean wand
throw towel
remove filter
dump grinds
wash filter
```

- one step per one line
- do not add “BEGIN” and “END”

# 1. Data Collection

## 1.1. Record One Session

1. The experimenter starts video recording on iPhone
2. The participant taps "start" on Apple Watch (timestamp is stored)
3. The participant claps to begin the task
4. The participant performs the task
5. The participant taps "stop" on Apple Watch (timestamp is stored)
6. The experimenter stops video recording on iPhone
7. The experimenter taps “audio-share” on Apple Watch to transfer the data to the phone

After the above, there will be the following files on the phone (filenames may be different):

- audio.wav
- motion.txt
- video.mp4

## 1.2. Annotate the Session

This annotation process assumes that `audio.wav` and `motion.txt` are synchronized and `video.mp4` is longer than them.

1. Prepare `steps.txt` as the set of labels used for annotation if you haven't.
2. Go to [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) and upload `video.mp4`.
3. Do annotation on the interface and download a CSV file (Rename it to `via_annotation.csv`).
    1. - Use the same label you defined in `steps.txt`. Do not use `"` in the step name.
    2. We care only the beginning timestamp; no need to precisely mark the end of each segment.
    3. Make sure to mark `clap`, which will be regarded as `BEGIN` later on.
    4. Make sure to mark `END`, which is usually right after the sensor recording stops.
    5. Delete the first 10 lines of the CSV file until `# CSV_HEADER=` such that we can load the CSV easily.
4. Listen to `audio.wav` and mark the timing of the clap, write it to `clap_time_audio.txt` (in ms).
    * If there is an end clap, you can write as `400.0,33000.0` in one line.
6. Gather files under `dataset/raw/{sid}/`. (Please also refer to Section 1.4)

## 1.3. Run Preprocessing Scripts

1. Run `preprocess_raw_dataset.py`, which will create `dataset/original/{sid}/`.
    1. `--task` to specify the task
    2. `--sid` to specify the session id
    3. `--no-video-export` to not export the video file (e.g., for privacy reason)
    4. `--no-sensor-input` to not use `audio.wav` and `motion.txt`. This is a special case and the subsequent process is not supported yet (2024.6). Please reach out to Riku for more details.
2. Run `check_dataset_format.py`, which will test whether the format of the task dataset is ok or not
    1. `--task` to specify the task

## 1.4. Structure the Dataset

Prepare `datadrive` directory wherever you want.

```
datadrive / tasks / {task_name}
│
└───dataset
    │   └───steps.txt
    │      
    └───raw (we create this at `1.2`)
    │   └───1
    │       └──-audio.wav
    │       └──-motion.txt
    │       └──-video.mp4
    │       └──-via_annotation_video.csv
    │       └──-clap_time_audio.txt
    │      
    └───original (this will be created at `1.3`)
    │   └───1
    │       └──-audio.wav
    │       └──-motion.txt
    │       └──-annotation.txt
    │      
    └───featurized (this will be created later in the HAR process)
            └──-1.pkl
```
