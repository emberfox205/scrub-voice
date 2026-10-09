---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: robot please clear the display please
- text: will you jump to frame 262 please
- text: robot call central supply for more sponges if you please
- text: operator dismiss the modality
- text: could you pitch up 10 degrees
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
- **Number of Classes:** 13 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label           | Examples                                                                                                                                                                                                                                                                                              |
|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| move_camera     | <ul><li>'will you camera right for the surgeon'</li><li>'would you kindly shift view backward by 25 centimetres now'</li><li>'can you shift view up by 9 cm if you please'</li></ul>                                                                                                                  |
| navigate_slice  | <ul><li>'could you please step forward 12 frames at once'</li><li>'will you display slice 332 safely'</li><li>'hey assistant previous slice right now'</li></ul>                                                                                                                                      |
| tilt_camera     | <ul><li>'hey assistant angle camera left by 20 degrees'</li><li>'would you kindly tilt left 8 degrees right now'</li><li>'surgeon needs to tilt down 8 degrees if you please'</li></ul>                                                                                                               |
| pick_tool       | <ul><li>'robot hand sterile tissue forceps for the surgeon'</li><li>'could you please ready a surgical forceps on monitor'</li><li>'assistant transfer sterile mosquito clamp'</li></ul>                                                                                                              |
| open_scan       | <ul><li>'launch patient x-ray for me'</li><li>'be sure to launch latest magnetic resonance as soon as possible'</li><li>'examine intraoperative coronal mri as soon as possible'</li></ul>                                                                                                            |
| reset_camera    | <ul><li>'please center view to home right away'</li><li>'reset laparoscope position on the screen'</li><li>'assistant please recenter endoscope to home'</li></ul>                                                                                                                                    |
| unknown         | <ul><li>'operator just talking to the nurse for inspection'</li><li>'i want to is the air conditioning working right now'</li><li>'turn up the operating room lights for inspection'</li></ul>                                                                                                        |
| zoom_display    | <ul><li>'could you magnify'</li><li>'assistant please zoom view thank you'</li><li>'please wider view right away'</li></ul>                                                                                                                                                                           |
| halt_emergency  | <ul><li>'could you emergency stop now'</li><li>'emergency brake hold as soon as possible'</li><li>'could you emergency abort at once please'</li></ul>                                                                                                                                                |
| close_scan      | <ul><li>'immediately dismiss tab at once'</li><li>'close the thoracic ct quickly'</li><li>'now minimize the active scan right now'</li></ul>                                                                                                                                                          |
| clear_emergency | <ul><li>'emergency retract to safe height right now'</li><li>'kindly emergency clearance right now for inspection'</li><li>'robot evacuate surgical field stat on primary viewport'</li></ul>                                                                                                         |
| log_event       | <ul><li>'could you please log event: hepatic artery clamped with note blood flow safely controlled as soon as possible'</li><li>'robot please document event: stapler fired'</li><li>'document milestone: bovie cautery check, comment adequate coagulation without active bleeders safely'</li></ul> |
| park_arm        | <ul><li>'park back ur10e arm outside surgical field'</li><li>'surgical assistant put away robotic arm to standby position'</li><li>'help me put away robotic manipulator away from patient on primary viewport'</li></ul>                                                                             |

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
preds = model("operator dismiss the modality")
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
| Word count   | 2   | 7.5049 | 19  |

| Label           | Training Sample Count |
|:----------------|:----------------------|
| clear_emergency | 588                   |
| close_scan      | 604                   |
| halt_emergency  | 565                   |
| log_event       | 621                   |
| move_camera     | 690                   |
| navigate_slice  | 714                   |
| open_scan       | 698                   |
| park_arm        | 569                   |
| pick_tool       | 703                   |
| reset_camera    | 512                   |
| tilt_camera     | 662                   |
| unknown         | 539                   |
| zoom_display    | 695                   |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (5, 5)
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
| Epoch  | Step  | Training Loss | Validation Loss |
|:------:|:-----:|:-------------:|:---------------:|
| 0.0005 | 1     | 0.1813        | -               |
| 0.0245 | 50    | 0.2170        | -               |
| 0.0490 | 100   | 0.2158        | -               |
| 0.0735 | 150   | 0.2103        | -               |
| 0.0980 | 200   | 0.2087        | -               |
| 0.1225 | 250   | 0.2047        | -               |
| 0.1471 | 300   | 0.1993        | -               |
| 0.1716 | 350   | 0.1829        | -               |
| 0.1961 | 400   | 0.1791        | -               |
| 0.2206 | 450   | 0.1660        | -               |
| 0.2451 | 500   | 0.1620        | -               |
| 0.2696 | 550   | 0.1519        | -               |
| 0.2941 | 600   | 0.1348        | -               |
| 0.3186 | 650   | 0.1271        | -               |
| 0.3431 | 700   | 0.1168        | -               |
| 0.3676 | 750   | 0.1086        | -               |
| 0.3922 | 800   | 0.0998        | -               |
| 0.4167 | 850   | 0.0860        | -               |
| 0.4412 | 900   | 0.0818        | -               |
| 0.4657 | 950   | 0.0772        | -               |
| 0.4902 | 1000  | 0.0834        | -               |
| 0.5147 | 1050  | 0.0666        | -               |
| 0.5392 | 1100  | 0.0590        | -               |
| 0.5637 | 1150  | 0.0613        | -               |
| 0.5882 | 1200  | 0.0562        | -               |
| 0.6127 | 1250  | 0.0535        | -               |
| 0.6373 | 1300  | 0.0535        | -               |
| 0.6618 | 1350  | 0.0491        | -               |
| 0.6863 | 1400  | 0.0431        | -               |
| 0.7108 | 1450  | 0.0406        | -               |
| 0.7353 | 1500  | 0.0405        | -               |
| 0.7598 | 1550  | 0.0331        | -               |
| 0.7843 | 1600  | 0.0321        | -               |
| 0.8088 | 1650  | 0.0355        | -               |
| 0.8333 | 1700  | 0.0364        | -               |
| 0.8578 | 1750  | 0.0334        | -               |
| 0.8824 | 1800  | 0.0295        | -               |
| 0.9069 | 1850  | 0.0270        | -               |
| 0.9314 | 1900  | 0.0300        | -               |
| 0.9559 | 1950  | 0.0292        | -               |
| 0.9804 | 2000  | 0.0271        | -               |
| 1.0    | 2040  | -             | 0.0190          |
| 1.0049 | 2050  | 0.0266        | -               |
| 1.0294 | 2100  | 0.0190        | -               |
| 1.0539 | 2150  | 0.0192        | -               |
| 1.0784 | 2200  | 0.0200        | -               |
| 1.1029 | 2250  | 0.0171        | -               |
| 1.1275 | 2300  | 0.0189        | -               |
| 1.1520 | 2350  | 0.0164        | -               |
| 1.1765 | 2400  | 0.0278        | -               |
| 1.2010 | 2450  | 0.0177        | -               |
| 1.2255 | 2500  | 0.0191        | -               |
| 1.25   | 2550  | 0.0185        | -               |
| 1.2745 | 2600  | 0.0173        | -               |
| 1.2990 | 2650  | 0.0206        | -               |
| 1.3235 | 2700  | 0.0162        | -               |
| 1.3480 | 2750  | 0.0142        | -               |
| 1.3725 | 2800  | 0.0151        | -               |
| 1.3971 | 2850  | 0.0160        | -               |
| 1.4216 | 2900  | 0.0135        | -               |
| 1.4461 | 2950  | 0.0155        | -               |
| 1.4706 | 3000  | 0.0132        | -               |
| 1.4951 | 3050  | 0.0115        | -               |
| 1.5196 | 3100  | 0.0103        | -               |
| 1.5441 | 3150  | 0.0114        | -               |
| 1.5686 | 3200  | 0.0115        | -               |
| 1.5931 | 3250  | 0.0096        | -               |
| 1.6176 | 3300  | 0.0112        | -               |
| 1.6422 | 3350  | 0.0098        | -               |
| 1.6667 | 3400  | 0.0089        | -               |
| 1.6912 | 3450  | 0.0088        | -               |
| 1.7157 | 3500  | 0.0088        | -               |
| 1.7402 | 3550  | 0.0077        | -               |
| 1.7647 | 3600  | 0.0064        | -               |
| 1.7892 | 3650  | 0.0066        | -               |
| 1.8137 | 3700  | 0.0056        | -               |
| 1.8382 | 3750  | 0.0062        | -               |
| 1.8627 | 3800  | 0.0056        | -               |
| 1.8873 | 3850  | 0.0052        | -               |
| 1.9118 | 3900  | 0.0042        | -               |
| 1.9363 | 3950  | 0.0056        | -               |
| 1.9608 | 4000  | 0.0054        | -               |
| 1.9853 | 4050  | 0.0041        | -               |
| 2.0    | 4080  | -             | 0.0015          |
| 2.0098 | 4100  | 0.0063        | -               |
| 2.0343 | 4150  | 0.0041        | -               |
| 2.0588 | 4200  | 0.0036        | -               |
| 2.0833 | 4250  | 0.0034        | -               |
| 2.1078 | 4300  | 0.0036        | -               |
| 2.1324 | 4350  | 0.0033        | -               |
| 2.1569 | 4400  | 0.0030        | -               |
| 2.1814 | 4450  | 0.0029        | -               |
| 2.2059 | 4500  | 0.0026        | -               |
| 2.2304 | 4550  | 0.0038        | -               |
| 2.2549 | 4600  | 0.0031        | -               |
| 2.2794 | 4650  | 0.0028        | -               |
| 2.3039 | 4700  | 0.0036        | -               |
| 2.3284 | 4750  | 0.0025        | -               |
| 2.3529 | 4800  | 0.0025        | -               |
| 2.3775 | 4850  | 0.0028        | -               |
| 2.4020 | 4900  | 0.0025        | -               |
| 2.4265 | 4950  | 0.0024        | -               |
| 2.4510 | 5000  | 0.0027        | -               |
| 2.4755 | 5050  | 0.0025        | -               |
| 2.5    | 5100  | 0.0022        | -               |
| 2.5245 | 5150  | 0.0024        | -               |
| 2.5490 | 5200  | 0.0028        | -               |
| 2.5735 | 5250  | 0.0021        | -               |
| 2.5980 | 5300  | 0.0023        | -               |
| 2.6225 | 5350  | 0.0019        | -               |
| 2.6471 | 5400  | 0.0025        | -               |
| 2.6716 | 5450  | 0.0022        | -               |
| 2.6961 | 5500  | 0.0021        | -               |
| 2.7206 | 5550  | 0.0023        | -               |
| 2.7451 | 5600  | 0.0018        | -               |
| 2.7696 | 5650  | 0.0024        | -               |
| 2.7941 | 5700  | 0.0019        | -               |
| 2.8186 | 5750  | 0.0021        | -               |
| 2.8431 | 5800  | 0.0018        | -               |
| 2.8676 | 5850  | 0.0019        | -               |
| 2.8922 | 5900  | 0.0018        | -               |
| 2.9167 | 5950  | 0.0018        | -               |
| 2.9412 | 6000  | 0.0023        | -               |
| 2.9657 | 6050  | 0.0018        | -               |
| 2.9902 | 6100  | 0.0017        | -               |
| 3.0    | 6120  | -             | 0.0006          |
| 3.0147 | 6150  | 0.0014        | -               |
| 3.0392 | 6200  | 0.0015        | -               |
| 3.0637 | 6250  | 0.0018        | -               |
| 3.0882 | 6300  | 0.0016        | -               |
| 3.1127 | 6350  | 0.0016        | -               |
| 3.1373 | 6400  | 0.0014        | -               |
| 3.1618 | 6450  | 0.0014        | -               |
| 3.1863 | 6500  | 0.0014        | -               |
| 3.2108 | 6550  | 0.0015        | -               |
| 3.2353 | 6600  | 0.0015        | -               |
| 3.2598 | 6650  | 0.0013        | -               |
| 3.2843 | 6700  | 0.0015        | -               |
| 3.3088 | 6750  | 0.0014        | -               |
| 3.3333 | 6800  | 0.0014        | -               |
| 3.3578 | 6850  | 0.0015        | -               |
| 3.3824 | 6900  | 0.0014        | -               |
| 3.4069 | 6950  | 0.0012        | -               |
| 3.4314 | 7000  | 0.0013        | -               |
| 3.4559 | 7050  | 0.0014        | -               |
| 3.4804 | 7100  | 0.0014        | -               |
| 3.5049 | 7150  | 0.0013        | -               |
| 3.5294 | 7200  | 0.0014        | -               |
| 3.5539 | 7250  | 0.0019        | -               |
| 3.5784 | 7300  | 0.0013        | -               |
| 3.6029 | 7350  | 0.0014        | -               |
| 3.6275 | 7400  | 0.0011        | -               |
| 3.6520 | 7450  | 0.0011        | -               |
| 3.6765 | 7500  | 0.0014        | -               |
| 3.7010 | 7550  | 0.0013        | -               |
| 3.7255 | 7600  | 0.0014        | -               |
| 3.75   | 7650  | 0.0013        | -               |
| 3.7745 | 7700  | 0.0011        | -               |
| 3.7990 | 7750  | 0.0013        | -               |
| 3.8235 | 7800  | 0.0012        | -               |
| 3.8480 | 7850  | 0.0012        | -               |
| 3.8725 | 7900  | 0.0013        | -               |
| 3.8971 | 7950  | 0.0014        | -               |
| 3.9216 | 8000  | 0.0012        | -               |
| 3.9461 | 8050  | 0.0011        | -               |
| 3.9706 | 8100  | 0.0012        | -               |
| 3.9951 | 8150  | 0.0011        | -               |
| 4.0    | 8160  | -             | 0.0004          |
| 4.0196 | 8200  | 0.0010        | -               |
| 4.0441 | 8250  | 0.0012        | -               |
| 4.0686 | 8300  | 0.0011        | -               |
| 4.0931 | 8350  | 0.0010        | -               |
| 4.1176 | 8400  | 0.0010        | -               |
| 4.1422 | 8450  | 0.0010        | -               |
| 4.1667 | 8500  | 0.0012        | -               |
| 4.1912 | 8550  | 0.0010        | -               |
| 4.2157 | 8600  | 0.0010        | -               |
| 4.2402 | 8650  | 0.0012        | -               |
| 4.2647 | 8700  | 0.0010        | -               |
| 4.2892 | 8750  | 0.0012        | -               |
| 4.3137 | 8800  | 0.0010        | -               |
| 4.3382 | 8850  | 0.0010        | -               |
| 4.3627 | 8900  | 0.0011        | -               |
| 4.3873 | 8950  | 0.0009        | -               |
| 4.4118 | 9000  | 0.0010        | -               |
| 4.4363 | 9050  | 0.0010        | -               |
| 4.4608 | 9100  | 0.0010        | -               |
| 4.4853 | 9150  | 0.0009        | -               |
| 4.5098 | 9200  | 0.0010        | -               |
| 4.5343 | 9250  | 0.0011        | -               |
| 4.5588 | 9300  | 0.0010        | -               |
| 4.5833 | 9350  | 0.0010        | -               |
| 4.6078 | 9400  | 0.0009        | -               |
| 4.6324 | 9450  | 0.0014        | -               |
| 4.6569 | 9500  | 0.0010        | -               |
| 4.6814 | 9550  | 0.0009        | -               |
| 4.7059 | 9600  | 0.0010        | -               |
| 4.7304 | 9650  | 0.0009        | -               |
| 4.7549 | 9700  | 0.0009        | -               |
| 4.7794 | 9750  | 0.0010        | -               |
| 4.8039 | 9800  | 0.0009        | -               |
| 4.8284 | 9850  | 0.0009        | -               |
| 4.8529 | 9900  | 0.0009        | -               |
| 4.8775 | 9950  | 0.0008        | -               |
| 4.9020 | 10000 | 0.0011        | -               |
| 4.9265 | 10050 | 0.0008        | -               |
| 4.9510 | 10100 | 0.0008        | -               |
| 4.9755 | 10150 | 0.0014        | -               |
| 5.0    | 10200 | 0.0018        | 0.0003          |

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