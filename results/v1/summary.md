# WasteVision results (2026-10-23)

Environment: {'tensorflow': '2.21.0', 'keras': '3.13.2', 'python': '3.13.15', 'gpu': ['Tesla T4']}

## Data
Totals by split: {'dwsd_test': 192, 'test': 1665, 'train': 8503, 'val': 1665}
By source: {'DWSD': 927, 'GD': 6397, 'TrashBox': 4701}
Raw images found: {'GD': 12259, 'TrashBox': 14279}
Excluded folders: {'TrashBox:medical/gloves': 355, 'TrashBox:medical/masks': 400, 'TrashBox:metal/construction scrap': 431, 'TrashBox:plastic/cigarette butt': 405}
Cleaning drops: {'unreadable (UnidentifiedImageError)': 2, 'unreadable (SyntaxError)': 1}
Deduplication: {'threshold': 5, 'images_before': 24944, 'images_after': 24348, 'duplicate_pairs': 617, 'exact_pairs': 441, 'cross_source_pairs': 36, 'clusters_with_conflicting_labels': 18}
Leakage audit (held-out near-duplicates of train): {'val': 0, 'test': 0, 'dwsd_test': 1}
DWSD: files {'images': 784, 'masks': 784, 'paired': 784}, mask formats {'index': 784}, mapping index values, photos 784, groups 441, crops by class {'aluminium-foil': 55, 'cloth': 7, 'glass': 31, 'metal-bottle': 31, 'nylon': 6, 'paper': 21, 'paper-cup': 17, 'plastic': 39, 'plastic-bottle': 2, 'plastic-container': 11, 'plastic-cup': 35, 'thermocol': 672}, skipped {'too_small': 1037, 'excluded_class': 202}, zip checksum ok: True

## Model comparison
| Model | Val macro-F1 | Test class acc | Test macro-F1 | Test stream acc | Stream acc (top-class lookup) | Special Care recall @0.20 | False alarm @0.20 | ECE test before → after | Temperature | Indian class acc [95% CI] | Indian stream acc [95% CI] | Params (M) | Size (MB) | Latency GPU (ms) | Latency CPU (ms) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scratch_cnn | 0.48 | 0.508 | 0.497 | 0.795 | 0.784 | 0.93 | 0.328 | 0.046 → 0.055 | 1.23 | 0.349 [0.255, 0.437] | 0.875 [0.824, 0.920] | 1.91 | 7.8 | 4.8 | 70.0 | 42.6 |
| mobilenetv2 | 0.873 | 0.846 | 0.844 | 0.936 | 0.932 | 0.949 | 0.076 | 0.022 → 0.020 | 1.03 | 0.776 [0.706, 0.845] | 0.995 [0.984, 1.000] | 2.42 | 10.3 | 6.3 | 40.7 | 8.6 |
| efficientnetv2b0 | 0.889 | 0.886 | 0.883 | 0.959 | 0.956 | 0.976 | 0.062 | 0.010 → 0.014 | 1.02 | 0.771 [0.697, 0.836] | 0.984 [0.960, 1.000] | 6.08 | 25.3 | 12.1 | 60.9 | 10.1 |

## Best model: efficientnetv2b0 (selected on validation macro-F1)
Temperature: 1.018; ECE val 0.015 -> 0.014; ECE test 0.010 -> 0.014
Test metrics: {"n": 1665, "class_acc": 0.8864864864864865, "macro_f1": 0.883077823456679, "stream_acc": 0.9585585585585585, "stream_acc_top_class": 0.9561561561561561, "sc_recall@0.33": 0.9563106796116505, "sc_false_alarm@0.33": 0.040702314445331206, "sc_recall@0.20": 0.9757281553398058, "sc_false_alarm@0.20": 0.06225059856344772, "sc_recall@0.10": 0.9830097087378641, "sc_false_alarm@0.10": 0.09816440542697526, "sc_recall_argmax": 0.9344660194174758}
Validation metrics: {"n": 1665, "class_acc": 0.893093093093093, "macro_f1": 0.8888195171277586, "stream_acc": 0.9717717717717718, "stream_acc_top_class": 0.9693693693693693, "sc_recall@0.33": 0.9661016949152542, "sc_false_alarm@0.33": 0.038338658146964855, "sc_recall@0.20": 0.9830508474576272, "sc_false_alarm@0.20": 0.054313099041533544, "sc_recall@0.10": 0.9854721549636803, "sc_false_alarm@0.10": 0.09185303514376997, "sc_recall_argmax": 0.9491525423728814}
Not-sure thresholds (from validation): stream 0.449, class 0.695; targets met: {'stream': True, 'cls': True}
Test selective stream: {'coverage': 0.9993993993993994, 'accuracy_answered': 0.9585336538461539}; selective class: {'coverage': 0.836036036036036, 'accuracy_answered': 0.9554597701149425}
Test accuracy by source: {'GD': {'n': 960, 'class_acc': 0.8927083333333333}, 'TrashBox': {'n': 705, 'class_acc': 0.8780141843971632}}
Indian (DWSD) held-out: {"n": 192, "class_acc": 0.7708333333333334, "macro_f1": 0.37407407407407406, "stream_acc": 0.984375, "stream_acc_top_class": 0.9791666666666666, "sc_recall@0.33": NaN, "sc_false_alarm@0.33": 0.03125, "sc_recall@0.20": NaN, "sc_false_alarm@0.20": 0.0625, "sc_recall@0.10": NaN, "sc_false_alarm@0.10": 0.10416666666666667, "sc_recall_argmax": NaN}; 95% CIs: {'class_acc': [0.696506438396254, 0.8358228090317642], 'stream_acc': [0.9601882827243094, 1.0], 'sc_false_alarm': [0.024271844660194174, 0.10551403130059363]}; groups: 71

## Per-class report (test)
```
                    precision    recall  f1-score   support

        biological      0.980     0.924     0.951       105
   paper_cardboard      0.870     0.894     0.882       180
           plastic      0.886     0.733     0.802       180
             metal      0.780     0.889     0.831       180
             glass      0.822     0.872     0.846       180
           clothes      0.967     0.972     0.970       180
             shoes      0.961     0.956     0.958       180
             trash      0.875     0.721     0.790        68
           battery      0.917     0.884     0.900       112
medicines_syringes      0.832     0.908     0.869       120
           e_waste      0.912     0.917     0.914       180

          accuracy                          0.886      1665
         macro avg      0.891     0.879     0.883      1665
      weighted avg      0.889     0.886     0.886      1665

```
