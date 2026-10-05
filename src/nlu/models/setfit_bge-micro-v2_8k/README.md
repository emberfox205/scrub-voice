---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Now zoom out double scale on primary monitor
- text: Magnify viewport by 1.5 times on axial view
- text: Now open chest X-ray
- text: Nurse please move viewport up 20 pixels on primary monitor
- text: Magnify active viewport on sagittal view for comparison
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
model-index:
- name: SetFit
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 1.0
      name: Accuracy
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 12 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label        | Examples                                                                                                                                                                                                                                                                                                                        |
|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VIEW_RESET   | <ul><li>'Reset viewport to default on primary monitor'</li><li>'Hey assistant reset viewer display orientation for coronal view for comparison'</li><li>'Reset viewer display orientation for sagittal view on screen'</li></ul>                                                                                                |
| ZOOM_OUT     | <ul><li>'Assistant zoom out three times on primary monitor'</li><li>'Zoom out to wide view on coronal view immediately'</li><li>'Zoom out two times for comparison'</li></ul>                                                                                                                                                   |
| IMAGE_OPEN   | <ul><li>'Now bring up head CT'</li><li>'Load up the brain MRI on primary monitor'</li><li>'Could you open patient preoperative CT on primary monitor'</li></ul>                                                                                                                                                                 |
| ZOOM_IN      | <ul><li>'Now zoom in 2 times on sagittal view immediately'</li><li>'Assistant zoom in five x now'</li><li>'Increase zoom to 3.5 times immediately'</li></ul>                                                                                                                                                                    |
| PAN          | <ul><li>'Hey assistant move viewport right 10 pixels on primary monitor'</li><li>'Nurse please move viewport left 200 pixels for comparison'</li><li>'Move viewport left 75 pixels'</li></ul>                                                                                                                                   |
| SLICE_PREV   | <ul><li>'Now go back 15 slices please'</li><li>'Could you step back to earlier slice for comparison'</li><li>'Nurse please go back one frame on the viewer'</li></ul>                                                                                                                                                           |
| LOG_EVENT    | <ul><li>'Hey assistant log milestone: cystic duct clamped, note: titanium clip placed securely for comparison'</li><li>'Log milestone biopsy taken, note frozen section sent to pathology for comparison'</li><li>'Now log milestone: patient insufflation started, note: intra-abdominal pressure at 12 mmHg please'</li></ul> |
| GOTO_SLICE   | <ul><li>'Hey assistant navigate to slice number 260 please'</li><li>'Now display slice 169 immediately'</li><li>'Assistant display slice 67'</li></ul>                                                                                                                                                                          |
| IMAGE_CLOSE  | <ul><li>'Hide the active workspace on screen'</li><li>'Could you dismiss this current scan'</li><li>'Assistant hide the medical viewport for comparison'</li></ul>                                                                                                                                                              |
| SLICE_NEXT   | <ul><li>'Assistant move forward 10 slices for comparison'</li><li>'Assistant advance 20 images ahead on screen'</li><li>'Hey assistant advance forward 10 slices'</li></ul>                                                                                                                                                     |
| OUT_OF_SCOPE | <ul><li>'Is the microphone unmuted on zoom'</li><li>'Hey assistant send the zoom link to pathology on screen'</li><li>'Nurse please The patient has no known drug allergies please'</li></ul>                                                                                                                                   |
| CONTRAST_SET | <ul><li>'Now adjust window level for lung'</li><li>'Could you change preset to bone contrast on the viewer'</li><li>'Nurse please adjust window level for bone on the viewer'</li></ul>                                                                                                                                         |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 1.0      |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Now open chest X-ray")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 2   | 7.6485 | 17  |

| Label        | Training Sample Count |
|:-------------|:----------------------|
| CONTRAST_SET | 804                   |
| GOTO_SLICE   | 651                   |
| IMAGE_CLOSE  | 465                   |
| IMAGE_OPEN   | 797                   |
| LOG_EVENT    | 488                   |
| OUT_OF_SCOPE | 331                   |
| PAN          | 1091                  |
| SLICE_NEXT   | 641                   |
| SLICE_PREV   | 642                   |
| VIEW_RESET   | 408                   |
| ZOOM_IN      | 970                   |
| ZOOM_OUT     | 712                   |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 2
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: True
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0005 | 1    | 0.2412        | -               |
| 0.025  | 50   | 0.2102        | -               |
| 0.05   | 100  | 0.1986        | -               |
| 0.075  | 150  | 0.1704        | -               |
| 0.1    | 200  | 0.1339        | -               |
| 0.125  | 250  | 0.1126        | -               |
| 0.15   | 300  | 0.0869        | -               |
| 0.175  | 350  | 0.0695        | -               |
| 0.2    | 400  | 0.0697        | -               |
| 0.225  | 450  | 0.0563        | -               |
| 0.25   | 500  | 0.0525        | -               |
| 0.275  | 550  | 0.0442        | -               |
| 0.3    | 600  | 0.0375        | -               |
| 0.325  | 650  | 0.0457        | -               |
| 0.35   | 700  | 0.0358        | -               |
| 0.375  | 750  | 0.0299        | -               |
| 0.4    | 800  | 0.0327        | -               |
| 0.425  | 850  | 0.0286        | -               |
| 0.45   | 900  | 0.0250        | -               |
| 0.475  | 950  | 0.0250        | -               |
| 0.5    | 1000 | 0.0196        | -               |
| 0.525  | 1050 | 0.0252        | -               |
| 0.55   | 1100 | 0.0232        | -               |
| 0.575  | 1150 | 0.0225        | -               |
| 0.6    | 1200 | 0.0191        | -               |
| 0.625  | 1250 | 0.0196        | -               |
| 0.65   | 1300 | 0.0190        | -               |
| 0.675  | 1350 | 0.0191        | -               |
| 0.7    | 1400 | 0.0158        | -               |
| 0.725  | 1450 | 0.0184        | -               |
| 0.75   | 1500 | 0.0156        | -               |
| 0.775  | 1550 | 0.0172        | -               |
| 0.8    | 1600 | 0.0160        | -               |
| 0.825  | 1650 | 0.0154        | -               |
| 0.85   | 1700 | 0.0142        | -               |
| 0.875  | 1750 | 0.0151        | -               |
| 0.9    | 1800 | 0.0137        | -               |
| 0.925  | 1850 | 0.0135        | -               |
| 0.95   | 1900 | 0.0163        | -               |
| 0.975  | 1950 | 0.0136        | -               |
| 1.0    | 2000 | 0.0136        | -               |

### Framework Versions
- Python: 3.13.5
- SetFit: 1.2.0
- Sentence Transformers: 6.1.0
- Transformers: 5.15.0.dev0
- PyTorch: 2.6.0+cu124
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->