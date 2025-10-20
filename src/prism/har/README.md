# PrISM --- HAR module

This is a frame-level human activity recognition (HAR). 
The backbone code comes from [SAMoSA](https://github.com/cmusmashlab/SAMoSA/tree/main).

# Scripts

## featurization.py

This will create frame-level multimodal features at `datadrive / tasks / {task_name} / dataset / featurized` by using data at `datadrive / tasks / {task_name} / dataset / original`. 

```
$ python featurization.py --task latte_making
```

- You can apply the featurization to a specific session by using `--sids`.

## idle_detection.py

This will update the above feature file by adding a `is_idle` key to indicate if each frame is idle or not.

```
$ python idle_detection.py --task latte_making
```

- You can apply the detection to a specific session by using `--sids`.

## classificattion.py

This will generate results for the frame-level step classification (Leave-One-Session-Out) at `datadrive / tasks / {task_name} / har / {model_hash}`.

```
$ python classification.py --task latte_making
```

- You can specify test sessions by using `--test_sids`.
- You can specify the number of training samples by using `--n_train`.
- You can specify whether to use the idle filter by using `--use_idle_filter`.
- You can specify the model hash by using `--model_hash`; the default is the timestamp.


# API

```
from prism.har import HumanActivityRecognitionAPI

model_hash = 'test_model'
har_api = HumanActivityRecognitionAPI(task_name='latte_making', model_hash=model_hash)
```