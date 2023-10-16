# Eliciting Human Preferences with Language Models
Paper link:

Abstract: Language models (LMs) can be directed to perform target tasks by using labeled examples or natural language prompts. But selecting examples or writing prompts for can be challenging--especially in tasks that involve unusual edge cases, demand precise articulation of nebulous preferences, or require an accurate mental model of LM behavior. We propose to use *LMs themselves* to guide the task specification process. In this paper, we introduce **Generative Active Task Elicitation (GATE)**: a learning framework in which models elicit and infer intended behavior through free-form, language-based interaction with users. We study GATE in three domains: email validation, content recommendation, and moral reasoning. In preregistered experiments, we show that LMs prompted to perform GATE (e.g., by generating open-ended questions or synthesizing informative edge cases) elicit responses that are often more informative than user-written prompts or labels. Users report that interactive task elicitation requires less effort than prompting or example labeling and surfaces novel considerations not initially anticipated by users. Our findings suggest that LM-driven elicitation can be a powerful tool for aligning models to complex human preferences and values.


## Setting up your environment
Run the following to set up your conda environment for running GATE:
```bash
conda create -n gen_al PYTHON=3.10
pip install -r requirements.txt
```

## Running the elicitation interface
To run the user interface that elicits humans for their preferences, use:
```bash
python WebInterface/server/webserver.py
```
The interface will randomly chooses a domain and query method (from among supervised learning, prompting, pool-based active learning, generative active learning, generative yes-or-no questions, generative open-ended questions) and saves the resulting transcript  under `annotations_gpt-4/`.

We are not releasing the human-model transcripts we collected at this time due to privacy concerns.

## Evaluating elicitation methods

Given a set of elicitation transcripts and gold human responses (produced by running the elicitation interface above), we can evaluate how well a model is able to make decisions by running the command:

```bash
python run_human_experiments.py
```

This prompts a language model to make decisions based on the contents of the transcript and compares them to the human-provided decisions on those same examples.

