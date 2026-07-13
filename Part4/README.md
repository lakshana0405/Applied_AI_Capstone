# Part 4 – LLM Powered Feature

## Selected Track

Track C – Model Prediction Explanation Pipeline

---

# Objective

The objective of this part is to combine the trained machine learning model from Part 3 with a Large Language Model (LLM) to generate structured explanations for predictions.

The LLM receives the feature values, predicted class, and predicted probability and returns a structured JSON explanation.

---

# LLM API

Provider

OpenRouter

Communication Library

requests

API Key

Stored securely using

```python
os.environ["LLM_API_KEY"]
```

No API key is hardcoded in the notebook.

---

# call_llm Function

A reusable function named

```
call_llm()
```

was implemented.

The function

- creates the JSON payload
- sends an HTTP POST request
- checks status code
- parses the JSON response
- returns only the generated message

---

# System Prompt

```
You are an AI assistant.

Output ONLY valid JSON.

Do not generate explanations outside JSON.
```

---

# User Prompt Template

```
Features:
{feature_values}

Predicted Class:
{prediction}

Probability:
{probability}
```

---

# Temperature

Temperature = 0

Reason

Temperature 0 produces deterministic outputs that are suitable for structured JSON generation.

Higher temperatures introduce randomness and may generate inconsistent JSON.

---

# Temperature Comparison

| Temperature | Behaviour |
|------------|-----------|
| 0 | Deterministic and consistent JSON |
| 0.7 | More diverse responses with increased variability |

---

# JSON Schema

The expected JSON contains

- prediction_label
- confidence_level
- top_reason
- second_reason
- next_step

All fields are validated using jsonschema.validate().

If validation fails, a fallback response is returned.

---

# PII Guardrail

Before every API request a regular expression checks for

- Email addresses
- Phone numbers

If PII is detected

```
Input blocked: PII detected.
```

is displayed and the API request is not sent.

---

# Guardrail Demonstration

| Input | Result |
|------|--------|
| lakshana@gmail.com | Blocked |
| Insurance customer information | Allowed |

---

# Model Prediction Pipeline

The following steps were executed.

1. Load best_model.pkl

2. Generate prediction

3. Generate prediction probability

4. Send prediction and feature values to the LLM

5. Parse JSON response

6. Validate JSON using jsonschema

7. Display explanation

---

# Demonstration

Three different feature vectors were evaluated.

For each input the notebook displays

- Feature values
- Predicted class
- Prediction probability
- LLM JSON explanation
- Validation result

---

# Files Included

- Part4.ipynb
- best_model.pkl
- README.md
- requirements.txt

---

# Conclusion

A complete LLM-powered prediction explanation pipeline was implemented.

The machine learning model generated predictions while the LLM converted those predictions into structured JSON explanations.

Schema validation ensured correctly formatted outputs, while the PII guardrail prevented sensitive information from being transmitted to the API.

This demonstrates how traditional machine learning models and Large Language Models can be combined into a production-oriented AI workflow.
