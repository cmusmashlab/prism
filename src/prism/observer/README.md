# PrISM --- Observer module

This is an intervention policy algorithm to enable proactive intervention from the assistant.

# Scripts

## calculate_remaining_time.py

This will generate the estimated remaining time `D_t` distribution over time at `datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} / remaining_time_distribution.pkl`.

```
$ python calculate_remaining_time.py --task latte_making --model_hash XXX
```

- You can specify test participants by using `--test_sids`.
- Each pickle file contains a dictionary with keys `expectations`, `entropies`, and `ground_truth`.

## evaluate_policy.py

This will generate the results for the intervention policy at `datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} /`.

```
$ python evaluate_policy.py --task latte_making --model_hash XXX
```

- You can specify test participants by using `--test_sids`.

# API

```
from prism.observer import ObserverAPI

# target_step: Step 1
policy_config = {
    target_step_index: {'h_threshold': 0.3, 'offset': 15}
}
observer_api = ObserverAPI(task_name='latte_making', policy_config=policy_config)
```
