# PrISM --- Tracker module

This is an extended Viterbi algorithm to postprocess the frame-level HAR outputs by leveraging a transition graph.

# Scripts

## tracking.py

This will generate post-processed results at `datadrive / tasks / {task_name} / tracker / {model_hash}`.

```
$ python tracking.py --task latte_making --model_hash XXX
```

- You can specify test sessions by using `--test_sids`.
- You need to specify the model hash that was used for training the HAR module.

# API

```
from prism.tracker import TrackerAPI

model_hash = 'test_model'
tracker_api = TrackerAPI(task_name='latte_making', model_hash=model_hash)
```