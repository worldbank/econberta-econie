# Economic Name Entity Recognition with Large Language Models

We've created EconBERTa, a new tool designed to understand and pull out specific information from economic research. It's trained to be really good at recognizing important bits in texts about economics studies. Nevertheless, the model needs a high amount of manually labeled data. Labelization made by experts which is really time consuming. We decided to test Large Language Models on the task in a zero or few shot learning paradigm. Here we present the development we made and the results we achieved.

## Few Shot Learning with openAI

Few-shot learning is a technique where a machine learning model learns to make accurate predictions from a very small amount of training data. Large language models are well-suited for this because they have been pre-trained on vast datasets, enabling them to generalize new tasks quickly with minimal examples.

At first, we perform the LLM entity detection at a *sentence level*. Meaning that we give the model a sentence and ask it to predict the entities in the sentence. We then compare the results with the previous EconBERTa model.

In this shot we decided to test the `GPT-3-turbo` first on the main task. We used the OpenAI API and all the functions are in `src/zsl/llm_utils.py`.

### Pipeline

The main pipeline follows a few steps:
1. We collect a few examples from the training set `train.conll` to get examples of entities for each label
2. We concatenate the examples with an instruction prompt
3. We collect the answers of the LLM and produce a token-level prediction that we compare with the previous EconBERTa

The main schema of the final prompt would be:

```markdown
### INSTRUCTION

My awesome instruction

### EXAMPLES

Text: my first example of sentence
A: {"my_label": "['my_first_entity', 'my_second_entity']"}

### INFERENCES

Text: my text to infer entities from
A:
```

**Examples**

To compute the examples, we only randomly pick some examples to get at least: one example without any entity in the sentence. Then for each of the label we pick one example that has at least this entity once, and we may have other entities too.

Then we provide those examples after the instructions as presented in previous schema. If you'd like to change the way the examples are displayed you can work on the `examples.prompt`. If you'd like to quit the examples you can do so when applying the llm, putting the `n` argument to 0. Then you will work on a zero shot learning paradigm.

**Prompt**

We computed in `instruction.prompt` file were you can modify the instruction of to spot entities. The main idea here is to present the definitions of the entities and give some general rules based from the annotation guidelines of the main task.

**Script**

*Convert ConLL to CSV*

We had to convert the `conll` data to get the whole sentences. that's why we created this script to go from `conll` to `csv`. Here is the script to apply:

```python
python src/zsl/conll_to_csv.py --path_data data/econ_ie --name_conll _train --name_csv _train
```

*Apply LLM*

If you'd like to apply the LLM with the zero/few-shot learning. You can use the following command:

```bash
python src/zsl/apply_llm.py --path_data data/econ_ie --example_csv_name train --path_prompt src/zsl --instruction_prompt instruction.prompt --examples_prompt examples.prompt --inference_prompt inference.prompt --inference_csv_name test --path_output data/econ_ie/20240419_ZSL.jsonl --verbose
```

*Compute Metrics*

If you'd like to compute the metrics from the new generated `jsonl` file from openAI you can use the following command:

```bash
python src/zsl/metrics.py --path_data_test data/econ_ie/test.csv --path_data_preds data/processed/20240417_FSL_LLM_responses.jsonl --path_settings src/zsl/settings_metrics.json
```
