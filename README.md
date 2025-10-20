# PrISM: Procedure Interaction from Sensing Module

This is a repository for the research code in which we aim to develop a real-time intelligent assistant that navigates users through dialogues during procedural tasks (e.g., cooking, latte-making, medical self-care).

<img width="1894" height="497" alt="image" src="https://github.com/user-attachments/assets/17c3c61d-5e52-4719-a506-312a365a22a3" />



[Project Page](https://rikky0611.github.io/projects/prism.html)

## Publications
The code is structured in a modular manner, from underlying sensing mechanisms to user interactions. We value your citation of the relevant publication.

- [PrISM-Tracker: A Framework for Multimodal Procedure Tracking Using Wearable Sensors and State Transition Information with User-Driven Handling of Errors and Uncertainty](https://rikky0611.github.io/resource/paper/prism-tracker_imwut2022_paper.pdf). Proceedings of the ACM on Interactive Mobile Wearable Ubiquitous Technology, Volume 6, Issue 4. (Ubicomp'23)
    - HAR and Tracker

- [PrISM-Observer: Intervention Agent to Help Users Perform Everyday Procedures Sensed using a Smartwatch](https://arxiv.org/abs/2407.16785). Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology (UIST'24), Pittsburgh, USA, Oct. 2024.
    - Observer (Proactive Intervention)

- [PrISM-Q&A: PrISM-Q&A: Step-Aware Voice Assistant on a Smartwatch Enabled by Multimodal Procedure Tracking and Large Language Models](https://rikky0611.github.io/resource/paper/prism-q&a_imwut2024_paper.pdf). Proceedings of the ACM on Interactive Mobile Wearable Ubiquitous Technology, Volume 8, Issue 4. (Ubicomp'25)
    - Question Answering (LLM augmented by Tracker)

- [Scaling Context-Aware Task Assistants that Learn from Demonstration and Adapt through Mixed-Initiative Dialogue](https://rikky0611.github.io/resource/paper/prism_uist2025_paper.pdf). Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology (UIST'25), Busan, South Korea, Sept. 2025.
     - Dialogue (Online Adaptation)

# What's in this repository now
For now, we have
- `data_collection`: smartwatch app + preprocess script
- `src`: modularized pipeline
    - HAR (frame-level human activity recognition)
    - Tracker (postprocess with an extended Viterbi algorithm)
    - Observer (proactive intervention based on the tracking result)
    - QA (question-answering interaction with an LLM augmented by sensing output)
    - Dialogue (context extraction from the dialogue with an LLM)

Real-time demo system is currently not public.

# Setup

Install the `prism` module into your environment

```
$ conda create -n "prism" python=3.10
$ conda activate prism
$ conda install --file requirements.txt -c conda-forge
```

After that, please run
```
$ python -m pip install -e src
```


Create a datadrive folder at your convenience. Make sure to update `PRISM_DATADRIVE`
```
$ export PRISM_DATADRIVE="path/to/datadrive"
```


In the datadrive, the structure will be
```
datadrive
│
└───pretrained_models
│   └───audio_model.h5
│   └───motion_model.h5
│   └───motion_norm_params.pkl
│  
└───tasks
    └───latte_making
          └───dataset
          └───har (will be generated)
          └───tracker (will be generated)
          └───...
```

Download the required files from the following links:
- pretrained_models: https://www.dropbox.com/sh/w3lo0f1k6w90b5w/AADuoDVSKuY9kQSPx2RRGJ_Ma?dl=0
- sample datasets: https://www.dropbox.com/sh/93jd6elugxgvm6k/AACL3XGiP8-UXPKIWK-h9Ud1a?dl=0
    - For now, there are `cooking` and `latte-making` tasks.
    - We are expanding the dataset with different tasks and additional interesting sensor sources. Stay tuned!

# License

This repository is published under MIT license. Please contact  Riku Arakawa and Mayank Goel if you would like another license for your use. 

# Contact

Feel free to contact [Riku Arakawa](mailto:rarakawa@andrew.cmu.edu) for any help, questions or general feedback!

# Acknowledgements
- Prasoon Patidar helped with the real-time demo.
- Hiromu Yakura helped with the implementation of the tracker and observer.
- Vimal Mollyn helped with the implementation of the HAR module.
- Suzanne Nie and Vicky Liu helped with the data collection pipeline.
