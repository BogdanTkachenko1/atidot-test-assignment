# Data Scientist - Assignment

## Context

Atidot works with life insurance policy data and builds predictive and decision-support systems for insurers. This assignment is designed to evaluate your ability to use AI tools effectively while still producing a clear, reproducible, and technically sound ML/AI artifact.

You are encouraged to use AI tools such as Cursor, ChatGPT, Claude, Copilot, or similar. We care less about whether you use AI and more about whether you can direct it intelligently, understand the result, and produce something that works.

The assignment is intended to take approximately 1-1.5 hours using AI assistance. A simple, correct, reproducible solution is better than an elaborate but fragile one.

## Inputs Provided

You will receive a repo skeleton containing:

```text
assignment/
  data/
    train_policies.csv
    segment_docs/
      segment_1.pdf
      segment_2.pdf
      segment_3.pdf
      segment_4.pdf
  src/
  README.md
```

The inputs are:

### Policy snapshot data

A CSV file resembling life insurance policy data used for lapse prediction. Each row represents a policy snapshot at a point in time.

### Segmentation documents

A small corpus of PDFs describing business/customer/policy segments, for example:

- high-income urban middle-aged policyholders
- post-retirement high-coverage policyholders
- low-premium early-tenure policies
- legacy policies with changing payment behavior

The exact segment definitions should be inferred from the PDFs and implemented in your segmentation engine.

## Required Deliverable

Submit a GitHub repository containing a runnable Python project.

Your project must implement both of the following commands:

```bash
python train.py --input data/train_policies.csv --output artifacts/
```

and:

```bash
python predict.py --input data/new_policies.csv --model artifacts/ --output predictions.csv
```

The `train.py` command should train the model and create all artifacts needed for later inference.

The `predict.py` command should load the saved artifacts and generate predictions on a new CSV without retraining.

## Required Functionality

Your solution should include two components.

### 1. Lapse Classifier

Build a simple model that predicts whether a policy will lapse within the next month. You may use any reasonable modeling approach, including but not limited to:

- random forest
- XGBoost
- LightGBM
- CatBoost
- sklearn pipelines

The model does not need to be powerful. We are more interested in whether the pipeline is valid, reproducible, and understandable.

Your solution should address:

- how the target is constructed
- which features are used
- how missing values are handled
- how categorical variables are handled
- how leakage is avoided
- how the model is evaluated

Use a reasonable validation method. A time-based split is preferred if the data supports it.

### 2. Segmentation Engine

Build a segmentation engine that assigns each policy snapshot to one of the business segments described in the PDFs.

You may implement this in any reasonable way, for example:

- manually extracted rules from the PDFs
- AI-assisted extraction of segment definitions followed by deterministic rules
- embedding-based matching between policy descriptions and segment descriptions
- a hybrid approach

For each policy, your segmentation engine should produce:

- a segment name
- a confidence score
- a short human-readable explanation

The segmentation does not need to be sophisticated. It does need to be explainable and reproducible.

## Required Output Format

The `predict.py` script must create a CSV file at the path specified by `--output`. The output file must include the following columns:

```text
policy_id
snapshot_date
lapse_probability_1m
lapse_rank
segment_name
segment_confidence
segment_reason
top_lapse_drivers
```

Where:

- `policy_id`: the policy identifier
- `snapshot_date`: the snapshot date used for prediction
- `lapse_probability_1m`: predicted probability of lapse within one month
- `lapse_rank`: relative risk ranking, with higher-risk policies ranked higher
- `segment_name`: assigned segment
- `segment_confidence`: confidence score for the segment assignment
- `segment_reason`: short explanation of why this segment was assigned
- `top_lapse_drivers`: short explanation of the main factors contributing to lapse risk

The explanations can be simple. You may use SHAP or feature importance if you want.

## Required Repo Structure

You may adjust the structure as needed, but your repo should be easy to understand. A good structure would be:

```text
assignment/
  data/
    train_policies.csv
    segment_docs/
      segment_1.pdf
      segment_2.pdf
      segment_3.pdf
      segment_4.pdf

  artifacts/
    # created by train.py

  src/
    features.py
    model.py
    segmentation.py
    utils.py

  train.py
  predict.py
  requirements.txt
  README.md
  MODEL_CARD.md
  AI_USAGE.md
```

You do not have to follow this structure exactly, but the two required commands must work.

## Required Documentation

Your submission must include the following files.

### README.md

Include:

- setup instructions
- how to train the model
- how to run predictions
- expected input format
- expected output format
- any assumptions you made

### MODEL_CARD.md

Include a short explanation of:

1. What the model predicts.
2. How the target was constructed.
3. What features were used.
4. What leakage risks you considered.
5. How validation was done.
6. How the segmentation engine works.
7. Known limitations.
8. What you would improve with more time.

### AI_USAGE.md

Include a short description of your AI-tool usage:

1. Which AI tools you used.
2. What you used them for.
3. Important prompts or prompt excerpts.
4. Which AI suggestions you accepted.
5. Which AI suggestions you rejected or modified.
6. Any parts of the solution you are uncertain about.

We do not expect a perfect prompt log. We want to understand how you used AI and whether you remained in control of the solution.

## Important Requirements

Your solution must be able to run on a new CSV with roughly the same schema as the training data. The new CSV may include realistic minor differences, such as:

- missing optional columns
- extra irrelevant columns
- null values
- unseen categorical values
- a different date range

Your inference pipeline should handle these gracefully where reasonable. The following should not be required during prediction:

- manual edits to the new CSV
- retraining the model
- re-running notebook cells
- changing hardcoded local paths

## Constraints

Please avoid:

- notebook solutions (notebooks are allowed for exploration)
- hardcoded absolute file paths
- solutions that only work on the provided training CSV
- using future information to predict past lapse
- relying on columns that directly reveal the target
- unexplainable segment assignment
- submitting AI-generated code you do not understand

## Evaluation Criteria

| Area | Weight |
|---|---:|
| Reproducibility and repo quality | 25% |
| Correct target construction and leakage awareness | 20% |
| Ability to run inference on a new CSV | 20% |
| Segmentation quality and explainability | 15% |
| Model evaluation sanity | 10% |
| AI usage maturity | 10% |

A simple solution that runs cleanly and is well understood will score better than a complex solution that is fragile or unclear.

## Follow-Up Interview

After submission, we will meet to review your work.

During the meeting, we may ask you to:

1. Explain your target construction.
2. Explain how you avoided leakage.
3. Explain how the segmentation engine works.
4. Run your pipeline scripts.
5. Debug minor schema or data issues.
6. Discuss what you would improve with more time.

We want to see whether you understand your own submission well.

## Expected Time

This assignment is designed to take approximately 1-1.5 hours with AI assistance.

Please do not over-engineer the solution. Focus on correctness, clarity, reproducibility, and practical judgment.
