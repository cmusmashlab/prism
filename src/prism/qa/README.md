# PrISM --- Q&A module

This is a question-answering (Q&A) algorithm to enable context-aware responses to the user question.

# Setup

You need OPENAI_API_KEY as we use their APIs as LLMs.

# Scripts

## evaluate.py

This will generate answers to the question dataset and evaluate them with automatic score.

```
$ python evaluate.py --task latte_making --model_hash XXX --llm gpt-4o-mini
```

- You can skip answer generation with `--eval_only`.

# Model
Note there is not API for this package. For details, refer to the dialogue package.

```
from prism.qa.algorithm import QuestionAnsweringModel

qa_model = QuestionAnsweringModel(task_name='latte_making', prompt_type='dialogue')
```
