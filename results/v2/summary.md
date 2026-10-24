# WasteVision results (2026-10-24)

Environment: {'tensorflow': '2.21.0', 'keras': '3.13.2', 'python': '3.13.16', 'gpu': ['Tesla T4']}

## Data
Sources downloaded: ['bepli', 'drinking', 'dwsd', 'gd', 'psu', 'realwaste', 'taco', 'tidy', 'trashbox', 'trashnet']; skipped: ['zerowaste']
Totals by split: {'dwsd_test': 192, 'test': 4069, 'train': 14750, 'val': 2028}
By source: {'BePLi': 780, 'DWSD': 927, 'Drinking': 1755, 'GD': 7638, 'RealWaste': 3574, 'TACO': 1674, 'TIDY': 177, 'TrashBox': 3072, 'TrashNet': 1442}
Mapped images per source and class (before dedup/cap): {'BePLi:plastic': 3730, 'Drinking:glass': 1224, 'Drinking:metal': 1060, 'Drinking:plastic': 2536, 'GD:battery': 756, 'GD:biological': 699, 'GD:clothes': 1892, 'GD:glass': 1736, 'GD:metal': 930, 'GD:paper_cardboard': 2747, 'GD:plastic': 1597, 'GD:shoes': 1449, 'GD:trash': 453, 'RealWaste:biological': 847, 'RealWaste:clothes': 318, 'RealWaste:glass': 420, 'RealWaste:metal': 790, 'RealWaste:paper_cardboard': 961, 'RealWaste:plastic': 921, 'RealWaste:trash': 495, 'TACO:battery': 2, 'TACO:biological': 8, 'TACO:glass': 182, 'TACO:metal': 474, 'TACO:paper_cardboard': 490, 'TACO:plastic': 2078, 'TACO:shoes': 7, 'TIDY:glass': 16, 'TIDY:metal': 8, 'TIDY:paper_cardboard': 63, 'TIDY:plastic': 90, 'TrashBox:e_waste': 2405, 'TrashBox:medicines_syringes': 810, 'TrashNet:glass': 1002, 'TrashNet:metal': 820, 'TrashNet:paper_cardboard': 1994, 'TrashNet:plastic': 964, 'TrashNet:trash': 274}
Share of real photos per split: {'dwsd_test': 1.0, 'test': 0.47554681739985255, 'train': 0.49098305084745764, 'val': 0.47337278106508873}
Raw images found: {'BePLi': 5932, 'Drinking': 9641, 'GD': 12259, 'RealWaste': 4752, 'TACO': 3241, 'TIDY': 304, 'TrashBox': 3970, 'TrashNet': 5054}
Excluded folders: {'Drinking:YOLO_imgs': 4821, 'TIDY:beach': 67, 'TIDY:street': 40, 'TIDY:zpersonal': 20, 'TrashBox:medical/gloves': 355, 'TrashBox:medical/masks': 400}
Cleaning drops: {'unreadable (SyntaxError)': 1}
Deduplication: {'threshold': 5, 'images_before': 37247, 'images_after': 31757, 'duplicate_pairs': 6336, 'exact_pairs': 6046, 'cross_source_pairs': 1510, 'clusters_with_conflicting_labels': 20}
Leakage audit (held-out near-duplicates of train): {'val': 0, 'test': 0, 'dwsd_test': 1}
DWSD: files {'images': 784, 'masks': 784, 'paired': 784}, mask formats {'index': 784}, mapping index values, photos 784, groups 441, crops by class {'aluminium-foil': 55, 'cloth': 7, 'glass': 31, 'metal-bottle': 31, 'nylon': 6, 'paper': 21, 'paper-cup': 17, 'plastic': 39, 'plastic-bottle': 2, 'plastic-container': 11, 'plastic-cup': 35, 'thermocol': 672}, skipped {'too_small': 1037, 'excluded_class': 202}, zip checksum ok: True

## Model comparison
| Model | Val macro-F1 | Test class acc | Test macro-F1 | Test stream acc | Stream acc (top-class lookup) | Special Care recall @0.20 | False alarm @0.20 | ECE test before → after | Temperature | Test class acc, real photos | Test macro-F1, real photos | Test class acc, web/mixed | Indian class acc [95% CI] | Indian stream acc [95% CI] | Params (M) | Size (MB) | Latency GPU (ms) | Latency CPU (ms) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scratch_cnn | 0.541 | 0.54 | 0.535 | 0.864 | 0.851 | 0.914 | 0.17 | 0.022 → 0.023 | 1.0 | 0.488 | 0.408 | 0.588 | 0.620 [0.549, 0.691] | 0.948 [0.915, 0.978] | 1.91 | 7.8 | 4.9 | 74.0 | 30.4 |
| mobilenetv2 | 0.845 | 0.842 | 0.847 | 0.968 | 0.966 | 0.979 | 0.042 | 0.012 → 0.014 | 0.99 | 0.765 | 0.704 | 0.911 | 0.786 [0.716, 0.850] | 0.995 [0.984, 1.000] | 2.42 | 10.3 | 5.5 | 47.8 | 11.7 |
| efficientnetv2b0 | 0.87 | 0.866 | 0.867 | 0.971 | 0.967 | 0.978 | 0.039 | 0.016 → 0.012 | 1.07 | 0.802 | 0.746 | 0.925 | 0.755 [0.678, 0.828] | 0.979 [0.953, 1.000] | 6.08 | 25.3 | 9.5 | 63.1 | 13.2 |

## Best model: efficientnetv2b0 (selected on validation macro-F1)
Temperature: 1.071; ECE val 0.018 -> 0.013; ECE test 0.016 -> 0.012
Test metrics: {"n": 4069, "class_acc": 0.8663062177439175, "macro_f1": 0.8668179029172262, "stream_acc": 0.9710002457606292, "stream_acc_top_class": 0.9668223150651266, "sc_recall@0.33": 0.9665809768637532, "sc_false_alarm@0.33": 0.02582801580066849, "sc_recall@0.20": 0.9781491002570694, "sc_false_alarm@0.20": 0.03859009419629292, "sc_recall@0.10": 0.9897172236503856, "sc_false_alarm@0.10": 0.05317532664843513, "sc_recall_argmax": 0.9575835475578406}
Validation metrics: {"n": 2028, "class_acc": 0.8722879684418146, "macro_f1": 0.8700679199384492, "stream_acc": 0.9758382642998028, "stream_acc_top_class": 0.9748520710059172, "sc_recall@0.33": 0.9812834224598931, "sc_false_alarm@0.33": 0.01995163240628779, "sc_recall@0.20": 0.9919786096256684, "sc_false_alarm@0.20": 0.026602176541717048, "sc_recall@0.10": 0.9946524064171123, "sc_false_alarm@0.10": 0.0562273276904474, "sc_recall_argmax": 0.9705882352941176}
Not-sure thresholds (from validation): stream 0.492, class 0.695; targets met: {'stream': True, 'cls': True}
Test selective stream: {'coverage': 0.9992627181125584, 'accuracy_answered': 0.9717166748647319}; selective class: {'coverage': 0.8058491029737036, 'accuracy_answered': 0.9478499542543458}
Test accuracy by source: {'GD': {'n': 1523, 'class_acc': 0.9251477347340775}, 'RealWaste': {'n': 719, 'class_acc': 0.8623087621696801}, 'TACO': {'n': 361, 'class_acc': 0.5678670360110804}, 'TrashBox': {'n': 611, 'class_acc': 0.9230769230769231}, 'TrashNet': {'n': 289, 'class_acc': 0.726643598615917}, 'TIDY': {'n': 35, 'class_acc': 0.6571428571428571}, 'Drinking': {'n': 350, 'class_acc': 0.9428571428571428}, 'BePLi': {'n': 181, 'class_acc': 0.9060773480662984}}
Test, real photos only: {"n": 1935, "class_acc": 0.8020671834625323, "macro_f1": 0.7460938753527985, "stream_acc": 0.9689922480620154, "stream_acc_top_class": 0.9607235142118863, "sc_recall@0.33": NaN, "sc_false_alarm@0.33": 0.028940568475452195, "sc_recall@0.20": NaN, "sc_false_alarm@0.20": 0.04237726098191214, "sc_recall@0.10": NaN, "sc_false_alarm@0.10": 0.05839793281653747, "sc_recall_argmax": NaN}
Test, web/mixed photos only: {"n": 2134, "class_acc": 0.9245548266166823, "macro_f1": 0.8762786421933282, "stream_acc": 0.9728209934395501, "stream_acc_top_class": 0.9723523898781631, "sc_recall@0.33": 0.9665809768637532, "sc_false_alarm@0.33": 0.021386430678466076, "sc_recall@0.20": 0.9781491002570694, "sc_false_alarm@0.20": 0.033185840707964605, "sc_recall@0.10": 0.9897172236503856, "sc_false_alarm@0.10": 0.045722713864306784, "sc_recall_argmax": 0.9575835475578406}
Indian (DWSD) held-out: {"n": 192, "class_acc": 0.7552083333333334, "macro_f1": 0.3452239789196311, "stream_acc": 0.9791666666666666, "stream_acc_top_class": 0.9791666666666666, "sc_recall@0.33": NaN, "sc_false_alarm@0.33": 0.026041666666666668, "sc_recall@0.20": NaN, "sc_false_alarm@0.20": 0.046875, "sc_recall@0.10": NaN, "sc_false_alarm@0.10": 0.08854166666666667, "sc_recall_argmax": NaN}; 95% CIs: {'class_acc': [0.6779440324858756, 0.8280279657643312], 'stream_acc': [0.9529396365876193, 1.0], 'sc_false_alarm': [0.01538265306122449, 0.08697773064687167]}; groups: 71

## Per-class report (test)
```
                    precision    recall  f1-score   support

        biological      0.905     0.981     0.941       311
   paper_cardboard      0.832     0.817     0.825       504
           plastic      0.832     0.665     0.739       529
             metal      0.784     0.885     0.831       487
             glass      0.864     0.888     0.876       517
           clothes      0.976     0.960     0.968       428
             shoes      0.983     0.983     0.983       292
             trash      0.735     0.673     0.703       223
           battery      0.987     0.922     0.954       167
medicines_syringes      0.734     0.857     0.791       161
           e_waste      0.903     0.947     0.924       450

          accuracy                          0.866      4069
         macro avg      0.867     0.871     0.867      4069
      weighted avg      0.867     0.866     0.865      4069

```

## All models (full numbers)
```json
{
 "scratch_cnn": {
  "temperature": 1.0044987574257347,
  "val": {
   "n": 2028,
   "class_acc": 0.5498027613412229,
   "macro_f1": 0.5405582624984363,
   "stream_acc": 0.8545364891518737,
   "stream_acc_top_class": 0.8496055226824457,
   "sc_recall@0.33": 0.8422459893048129,
   "sc_false_alarm@0.33": 0.10580411124546554,
   "sc_recall@0.20": 0.9090909090909091,
   "sc_false_alarm@0.20": 0.17412333736396615,
   "sc_recall@0.10": 0.9598930481283422,
   "sc_false_alarm@0.10": 0.2642079806529625,
   "sc_recall_argmax": 0.7192513368983957
  },
  "test": {
   "n": 4069,
   "class_acc": 0.5404276234947162,
   "macro_f1": 0.5353219706659933,
   "stream_acc": 0.8643401327107397,
   "stream_acc_top_class": 0.8510690587367904,
   "sc_recall@0.33": 0.8560411311053985,
   "sc_false_alarm@0.33": 0.10513521725919174,
   "sc_recall@0.20": 0.9138817480719794,
   "sc_false_alarm@0.20": 0.1701610452749924,
   "sc_recall@0.10": 0.9460154241645244,
   "sc_false_alarm@0.10": 0.2458219386204801,
   "sc_recall_argmax": 0.7339331619537275
  },
  "dwsd_test": {
   "n": 192,
   "class_acc": 0.6197916666666666,
   "macro_f1": 0.24857752489331436,
   "stream_acc": 0.9479166666666666,
   "stream_acc_top_class": 0.8854166666666666,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.140625,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.2708333333333333,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.5052083333333334,
   "sc_recall_argmax": NaN
  },
  "ece_val_before": 0.0281465311756971,
  "ece_val_after": 0.029312071274723517,
  "ece_test_before": 0.022348815970293118,
  "ece_test_after": 0.023158619831721072,
  "not_sure_stream_threshold": 0.9183934926986694,
  "not_sure_class_threshold": 0.8885236978530884,
  "not_sure_targets_met_on_val": {
   "stream": true,
   "cls": true
  },
  "test_selective_stream": {
   "coverage": 0.5362496927992135,
   "accuracy_answered": 0.9715857011915674
  },
  "test_selective_class": {
   "coverage": 0.08380437453919883,
   "accuracy_answered": 0.9354838709677419
  },
  "test_real": {
   "n": 1935,
   "class_acc": 0.4878552971576227,
   "macro_f1": 0.40787420329589624,
   "stream_acc": 0.9204134366925064,
   "stream_acc_top_class": 0.8770025839793282,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.08165374677002583,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.1400516795865633,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.2020671834625323,
   "sc_recall_argmax": NaN
  },
  "test_web": {
   "n": 2134,
   "class_acc": 0.5880974695407685,
   "macro_f1": 0.48109789685856386,
   "stream_acc": 0.8134957825679475,
   "stream_acc_top_class": 0.8275538894095595,
   "sc_recall@0.33": 0.8560411311053985,
   "sc_false_alarm@0.33": 0.13864306784660768,
   "sc_recall@0.20": 0.9138817480719794,
   "sc_false_alarm@0.20": 0.21312684365781712,
   "sc_recall@0.10": 0.9460154241645244,
   "sc_false_alarm@0.10": 0.308259587020649,
   "sc_recall_argmax": 0.7339331619537275
  },
  "test_by_source": {
   "GD": {
    "n": 1523,
    "class_acc": 0.562048588312541
   },
   "RealWaste": {
    "n": 719,
    "class_acc": 0.5132127955493742
   },
   "TACO": {
    "n": 361,
    "class_acc": 0.23545706371191136
   },
   "TrashBox": {
    "n": 611,
    "class_acc": 0.6530278232405892
   },
   "TrashNet": {
    "n": 289,
    "class_acc": 0.44982698961937717
   },
   "TIDY": {
    "n": 35,
    "class_acc": 0.2
   },
   "Drinking": {
    "n": 350,
    "class_acc": 0.6342857142857142
   },
   "BePLi": {
    "n": 181,
    "class_acc": 0.7237569060773481
   }
  },
  "dwsd_ci": {
   "class_acc": [
    0.5485577078288943,
    0.6908282115486759
   ],
   "stream_acc": [
    0.915171447568389,
    0.9776567349472377
   ],
   "sc_false_alarm": [
    0.1954009623095429,
    0.35682385918029447
   ]
  },
  "dwsd_groups": 71,
  "params_m": 1.911403,
  "size_mb": 7.802054,
  "train_minutes": 30.42129611968994,
  "latency_ms_gpu": 4.86871963333518,
  "latency_ms_cpu": 74.03717760001503
 },
 "mobilenetv2": {
  "temperature": 0.9875665455366804,
  "val": {
   "n": 2028,
   "class_acc": 0.8422090729783037,
   "macro_f1": 0.8449626510499519,
   "stream_acc": 0.9718934911242604,
   "stream_acc_top_class": 0.9664694280078896,
   "sc_recall@0.33": 0.9679144385026738,
   "sc_false_alarm@0.33": 0.022370012091898428,
   "sc_recall@0.20": 0.9812834224598931,
   "sc_false_alarm@0.20": 0.039298669891172915,
   "sc_recall@0.10": 0.9893048128342246,
   "sc_false_alarm@0.10": 0.05804111245465538,
   "sc_recall_argmax": 0.9572192513368984
  },
  "test": {
   "n": 4069,
   "class_acc": 0.8417301548291963,
   "macro_f1": 0.8474267590714402,
   "stream_acc": 0.9675595969525682,
   "stream_acc_top_class": 0.9660850331776849,
   "sc_recall@0.33": 0.9665809768637532,
   "sc_false_alarm@0.33": 0.027043451838347006,
   "sc_recall@0.20": 0.9794344473007712,
   "sc_false_alarm@0.20": 0.04223640230932847,
   "sc_recall@0.10": 0.9884318766066839,
   "sc_false_alarm@0.10": 0.0674567000911577,
   "sc_recall_argmax": 0.9485861182519281
  },
  "dwsd_test": {
   "n": 192,
   "class_acc": 0.7864583333333334,
   "macro_f1": 0.2848019610372855,
   "stream_acc": 0.9947916666666666,
   "stream_acc_top_class": 0.9895833333333334,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.026041666666666668,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.041666666666666664,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.0625,
   "sc_recall_argmax": NaN
  },
  "ece_val_before": 0.026719691617601737,
  "ece_val_after": 0.02687736464704752,
  "ece_test_before": 0.011751944502702522,
  "ece_test_after": 0.014212012507019575,
  "not_sure_stream_threshold": 0.5016489028930664,
  "not_sure_class_threshold": 0.6958544254302979,
  "not_sure_targets_met_on_val": {
   "stream": true,
   "cls": true
  },
  "test_selective_stream": {
   "coverage": 0.9995084787417056,
   "accuracy_answered": 0.9677895254487338
  },
  "test_selective_class": {
   "coverage": 0.7746375030720078,
   "accuracy_answered": 0.9378172588832487
  },
  "test_real": {
   "n": 1935,
   "class_acc": 0.7648578811369509,
   "macro_f1": 0.7041054757945399,
   "stream_acc": 0.9627906976744186,
   "stream_acc_top_class": 0.958139534883721,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.03772609819121447,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.056330749354005165,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.082687338501292,
   "sc_recall_argmax": NaN
  },
  "test_web": {
   "n": 2134,
   "class_acc": 0.9114339268978444,
   "macro_f1": 0.8677613880802041,
   "stream_acc": 0.971883786316776,
   "stream_acc_top_class": 0.9732895970009372,
   "sc_recall@0.33": 0.9665809768637532,
   "sc_false_alarm@0.33": 0.011799410029498525,
   "sc_recall@0.20": 0.9794344473007712,
   "sc_false_alarm@0.20": 0.022123893805309734,
   "sc_recall@0.10": 0.9884318766066839,
   "sc_false_alarm@0.10": 0.045722713864306784,
   "sc_recall_argmax": 0.9485861182519281
  },
  "test_by_source": {
   "GD": {
    "n": 1523,
    "class_acc": 0.9133289560078792
   },
   "RealWaste": {
    "n": 719,
    "class_acc": 0.8052851182197497
   },
   "TACO": {
    "n": 361,
    "class_acc": 0.5235457063711911
   },
   "TrashBox": {
    "n": 611,
    "class_acc": 0.9067103109656302
   },
   "TrashNet": {
    "n": 289,
    "class_acc": 0.7439446366782007
   },
   "TIDY": {
    "n": 35,
    "class_acc": 0.5142857142857142
   },
   "Drinking": {
    "n": 350,
    "class_acc": 0.9
   },
   "BePLi": {
    "n": 181,
    "class_acc": 0.9060773480662984
   }
  },
  "dwsd_ci": {
   "class_acc": [
    0.7164086972544683,
    0.8500320512820513
   ],
   "stream_acc": [
    0.9838626709146423,
    1.0
   ],
   "sc_false_alarm": [
    0.01657230890656305,
    0.07362124636745238
   ]
  },
  "dwsd_groups": 71,
  "params_m": 2.423371,
  "size_mb": 10.29156,
  "train_minutes": 11.652334642410278,
  "latency_ms_gpu": 5.53181376668969,
  "latency_ms_cpu": 47.825991966662215
 },
 "efficientnetv2b0": {
  "temperature": 1.0713930060008507,
  "val": {
   "n": 2028,
   "class_acc": 0.8722879684418146,
   "macro_f1": 0.8700679199384492,
   "stream_acc": 0.9758382642998028,
   "stream_acc_top_class": 0.9748520710059172,
   "sc_recall@0.33": 0.9812834224598931,
   "sc_false_alarm@0.33": 0.01995163240628779,
   "sc_recall@0.20": 0.9919786096256684,
   "sc_false_alarm@0.20": 0.026602176541717048,
   "sc_recall@0.10": 0.9946524064171123,
   "sc_false_alarm@0.10": 0.0562273276904474,
   "sc_recall_argmax": 0.9705882352941176
  },
  "test": {
   "n": 4069,
   "class_acc": 0.8663062177439175,
   "macro_f1": 0.8668179029172262,
   "stream_acc": 0.9710002457606292,
   "stream_acc_top_class": 0.9668223150651266,
   "sc_recall@0.33": 0.9665809768637532,
   "sc_false_alarm@0.33": 0.02582801580066849,
   "sc_recall@0.20": 0.9781491002570694,
   "sc_false_alarm@0.20": 0.03859009419629292,
   "sc_recall@0.10": 0.9897172236503856,
   "sc_false_alarm@0.10": 0.05317532664843513,
   "sc_recall_argmax": 0.9575835475578406
  },
  "dwsd_test": {
   "n": 192,
   "class_acc": 0.7552083333333334,
   "macro_f1": 0.3452239789196311,
   "stream_acc": 0.9791666666666666,
   "stream_acc_top_class": 0.9791666666666666,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.026041666666666668,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.046875,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.08854166666666667,
   "sc_recall_argmax": NaN
  },
  "ece_val_before": 0.017825448602673086,
  "ece_val_after": 0.01325505293714693,
  "ece_test_before": 0.015533739744401115,
  "ece_test_after": 0.011923129328142757,
  "not_sure_stream_threshold": 0.49206581711769104,
  "not_sure_class_threshold": 0.6949189901351929,
  "not_sure_targets_met_on_val": {
   "stream": true,
   "cls": true
  },
  "test_selective_stream": {
   "coverage": 0.9992627181125584,
   "accuracy_answered": 0.9717166748647319
  },
  "test_selective_class": {
   "coverage": 0.8058491029737036,
   "accuracy_answered": 0.9478499542543458
  },
  "test_real": {
   "n": 1935,
   "class_acc": 0.8020671834625323,
   "macro_f1": 0.7460938753527985,
   "stream_acc": 0.9689922480620154,
   "stream_acc_top_class": 0.9607235142118863,
   "sc_recall@0.33": NaN,
   "sc_false_alarm@0.33": 0.028940568475452195,
   "sc_recall@0.20": NaN,
   "sc_false_alarm@0.20": 0.04237726098191214,
   "sc_recall@0.10": NaN,
   "sc_false_alarm@0.10": 0.05839793281653747,
   "sc_recall_argmax": NaN
  },
  "test_web": {
   "n": 2134,
   "class_acc": 0.9245548266166823,
   "macro_f1": 0.8762786421933282,
   "stream_acc": 0.9728209934395501,
   "stream_acc_top_class": 0.9723523898781631,
   "sc_recall@0.33": 0.9665809768637532,
   "sc_false_alarm@0.33": 0.021386430678466076,
   "sc_recall@0.20": 0.9781491002570694,
   "sc_false_alarm@0.20": 0.033185840707964605,
   "sc_recall@0.10": 0.9897172236503856,
   "sc_false_alarm@0.10": 0.045722713864306784,
   "sc_recall_argmax": 0.9575835475578406
  },
  "test_by_source": {
   "GD": {
    "n": 1523,
    "class_acc": 0.9251477347340775
   },
   "RealWaste": {
    "n": 719,
    "class_acc": 0.8623087621696801
   },
   "TACO": {
    "n": 361,
    "class_acc": 0.5678670360110804
   },
   "TrashBox": {
    "n": 611,
    "class_acc": 0.9230769230769231
   },
   "TrashNet": {
    "n": 289,
    "class_acc": 0.726643598615917
   },
   "TIDY": {
    "n": 35,
    "class_acc": 0.6571428571428571
   },
   "Drinking": {
    "n": 350,
    "class_acc": 0.9428571428571428
   },
   "BePLi": {
    "n": 181,
    "class_acc": 0.9060773480662984
   }
  },
  "dwsd_ci": {
   "class_acc": [
    0.6779440324858756,
    0.8280279657643312
   ],
   "stream_acc": [
    0.9529396365876193,
    1.0
   ],
   "sc_false_alarm": [
    0.01538265306122449,
    0.08697773064687167
   ]
  },
  "dwsd_groups": 71,
  "params_m": 6.084699,
  "size_mb": 25.306514,
  "train_minutes": 13.182688931624094,
  "latency_ms_gpu": 9.525530233319538,
  "latency_ms_cpu": 63.08578230000421
 }
}
```