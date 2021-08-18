# AdaGNN


Open-source code for ''AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter''.

## Environment
Experiments are carried out on a Titan RTX with Cuda 10.1.

## usage
Default parameter settings are with an 8-layered AdaGNN-S model.
Use as
```
python main.py
```
Pre-trained models on BlogCatalog can be found in pre_trained_examples.

Configure as you like:
```
python main.py --layers 8 --mode s --dataset BlogCatalog --hidden 128 --dropout 0.1
```
Log example:

```
Loading BlogCatalog dataset...
Dataset has 5196 nodes, 8189 features.
AdaGNN(
  (should_train_1): Adagnn_with_weight (8189 -> 128)
  (hidden_layers): ModuleList(
    (0): Adagnn_without_weight (128 -> 128)
    (1): Adagnn_without_weight (128 -> 128)
    (2): Adagnn_without_weight (128 -> 128)
    (3): Adagnn_without_weight (128 -> 128)
    (4): Adagnn_without_weight (128 -> 128)
    (5): Adagnn_without_weight (128 -> 128)
  )
  (should_train_2): Adagnn_with_weight (128 -> 6)
)
Epoch: 0001 loss_train: 1.8881 acc_train: 0.2351 loss_val: 1.8897 acc_val: 0.2223 time: 1.0622s
Epoch: 0002 loss_train: 5.6176 acc_train: 0.1753 loss_val: 5.8035 acc_val: 0.1598 time: 0.2379s
Epoch: 0003 loss_train: 2.8776 acc_train: 0.2062 loss_val: 2.8626 acc_val: 0.1983 time: 0.2111s
Epoch: 0004 loss_train: 2.9220 acc_train: 0.2293 loss_val: 2.8237 acc_val: 0.2628 time: 0.2111s
Epoch: 0005 loss_train: 2.2331 acc_train: 0.3372 loss_val: 2.1757 acc_val: 0.3831 time: 0.2190s
Epoch: 0006 loss_train: 1.8275 acc_train: 0.2466 loss_val: 1.8105 acc_val: 0.2339 time: 0.2160s
Epoch: 0007 loss_train: 1.6117 acc_train: 0.3642 loss_val: 1.6517 acc_val: 0.3696 time: 0.2151s
Epoch: 0008 loss_train: 1.5870 acc_train: 0.3083 loss_val: 1.6172 acc_val: 0.3090 time: 0.2170s
Epoch: 0009 loss_train: 1.5674 acc_train: 0.3314 loss_val: 1.6132 acc_val: 0.3215 time: 0.2180s
Epoch: 0010 loss_train: 1.5117 acc_train: 0.3622 loss_val: 1.5536 acc_val: 0.3580 time: 0.2230s
Epoch: 0011 loss_train: 1.4489 acc_train: 0.4451 loss_val: 1.4810 acc_val: 0.4216 time: 0.2250s
Epoch: 0012 loss_train: 1.4243 acc_train: 0.4663 loss_val: 1.4533 acc_val: 0.4629 time: 0.2170s
Epoch: 0013 loss_train: 1.4085 acc_train: 0.5145 loss_val: 1.4267 acc_val: 0.4697 time: 0.2210s
Epoch: 0014 loss_train: 1.3816 acc_train: 0.4701 loss_val: 1.4083 acc_val: 0.4524 time: 0.2190s
Epoch: 0015 loss_train: 1.3297 acc_train: 0.5183 loss_val: 1.3628 acc_val: 0.4889 time: 0.2200s
Epoch: 0016 loss_train: 1.2997 acc_train: 0.5222 loss_val: 1.3468 acc_val: 0.4928 time: 0.2210s
Epoch: 0017 loss_train: 1.2826 acc_train: 0.5318 loss_val: 1.3250 acc_val: 0.4937 time: 0.2210s
Epoch: 0018 loss_train: 1.2185 acc_train: 0.5491 loss_val: 1.2705 acc_val: 0.5294 time: 0.2200s
Epoch: 0019 loss_train: 1.1873 acc_train: 0.5645 loss_val: 1.2240 acc_val: 0.5525 time: 0.2330s
Epoch: 0020 loss_train: 1.1637 acc_train: 0.5723 loss_val: 1.2038 acc_val: 0.5332 time: 0.2170s
Epoch: 0021 loss_train: 1.1413 acc_train: 0.5703 loss_val: 1.1823 acc_val: 0.5476 time: 0.2280s
Epoch: 0022 loss_train: 1.0945 acc_train: 0.6108 loss_val: 1.1679 acc_val: 0.5630 time: 0.2240s
Epoch: 0023 loss_train: 1.0855 acc_train: 0.6397 loss_val: 1.1598 acc_val: 0.6256 time: 0.2200s
Epoch: 0024 loss_train: 1.0447 acc_train: 0.6532 loss_val: 1.0986 acc_val: 0.6487 time: 0.2180s
Epoch: 0025 loss_train: 1.0415 acc_train: 0.6455 loss_val: 1.1020 acc_val: 0.6179 time: 0.2151s
Epoch: 0026 loss_train: 1.0104 acc_train: 0.6551 loss_val: 1.0747 acc_val: 0.6410 time: 0.2170s
Epoch: 0027 loss_train: 0.9543 acc_train: 0.6975 loss_val: 1.0533 acc_val: 0.6526 time: 0.2270s
Epoch: 0028 loss_train: 0.9713 acc_train: 0.6821 loss_val: 1.0619 acc_val: 0.6583 time: 0.2131s
Epoch: 0029 loss_train: 0.9118 acc_train: 0.6975 loss_val: 1.0312 acc_val: 0.6535 time: 0.2340s
Epoch: 0030 loss_train: 0.8880 acc_train: 0.7033 loss_val: 0.9769 acc_val: 0.6622 time: 0.2200s
Epoch: 0031 loss_train: 0.8688 acc_train: 0.7013 loss_val: 0.9882 acc_val: 0.6660 time: 0.2200s
Epoch: 0032 loss_train: 0.8376 acc_train: 0.7283 loss_val: 0.9607 acc_val: 0.6805 time: 0.2260s
Epoch: 0033 loss_train: 0.8222 acc_train: 0.7457 loss_val: 0.9309 acc_val: 0.6891 time: 0.2200s
Epoch: 0034 loss_train: 0.7958 acc_train: 0.7553 loss_val: 0.9339 acc_val: 0.6824 time: 0.2250s
Epoch: 0035 loss_train: 0.7468 acc_train: 0.7765 loss_val: 0.9172 acc_val: 0.6968 time: 0.2300s
Epoch: 0036 loss_train: 0.7524 acc_train: 0.7534 loss_val: 0.9215 acc_val: 0.6805 time: 0.2210s
Epoch: 0037 loss_train: 0.7004 acc_train: 0.7611 loss_val: 0.9079 acc_val: 0.6853 time: 0.2160s
Epoch: 0038 loss_train: 0.6658 acc_train: 0.7958 loss_val: 0.8722 acc_val: 0.6997 time: 0.2200s
Epoch: 0039 loss_train: 0.6449 acc_train: 0.7842 loss_val: 0.8399 acc_val: 0.7122 time: 0.2370s
Epoch: 0040 loss_train: 0.6287 acc_train: 0.7900 loss_val: 0.8385 acc_val: 0.7093 time: 0.2290s
Epoch: 0041 loss_train: 0.6093 acc_train: 0.8112 loss_val: 0.8232 acc_val: 0.7344 time: 0.2220s
Epoch: 0042 loss_train: 0.5808 acc_train: 0.8227 loss_val: 0.8277 acc_val: 0.7141 time: 0.2160s
Epoch: 0043 loss_train: 0.5540 acc_train: 0.8304 loss_val: 0.8263 acc_val: 0.7315 time: 0.2220s
Epoch: 0044 loss_train: 0.5190 acc_train: 0.8266 loss_val: 0.7588 acc_val: 0.7488 time: 0.2330s
Epoch: 0045 loss_train: 0.4883 acc_train: 0.8690 loss_val: 0.7248 acc_val: 0.7449 time: 0.2350s
Epoch: 0046 loss_train: 0.4601 acc_train: 0.8748 loss_val: 0.7270 acc_val: 0.7517 time: 0.2210s
Epoch: 0047 loss_train: 0.4595 acc_train: 0.8709 loss_val: 0.7423 acc_val: 0.7565 time: 0.2160s
Epoch: 0048 loss_train: 0.4347 acc_train: 0.8767 loss_val: 0.6984 acc_val: 0.7767 time: 0.2270s
Epoch: 0049 loss_train: 0.4055 acc_train: 0.8882 loss_val: 0.7285 acc_val: 0.7565 time: 0.2131s
Epoch: 0050 loss_train: 0.4147 acc_train: 0.8748 loss_val: 0.6816 acc_val: 0.7498 time: 0.2340s
Epoch: 0051 loss_train: 0.3548 acc_train: 0.9094 loss_val: 0.6996 acc_val: 0.7748 time: 0.3863s
Epoch: 0052 loss_train: 0.3655 acc_train: 0.8844 loss_val: 0.6941 acc_val: 0.7757 time: 0.2788s
Epoch: 0053 loss_train: 0.3465 acc_train: 0.9056 loss_val: 0.6217 acc_val: 0.7825 time: 0.2439s
Epoch: 0054 loss_train: 0.3351 acc_train: 0.9171 loss_val: 0.6754 acc_val: 0.7911 time: 0.2121s
Epoch: 0055 loss_train: 0.2887 acc_train: 0.9268 loss_val: 0.6178 acc_val: 0.7844 time: 0.2220s
Epoch: 0056 loss_train: 0.2974 acc_train: 0.9133 loss_val: 0.6816 acc_val: 0.7892 time: 0.2131s
Epoch: 0057 loss_train: 0.3093 acc_train: 0.9133 loss_val: 0.6516 acc_val: 0.7940 time: 0.2160s
Epoch: 0058 loss_train: 0.2823 acc_train: 0.9094 loss_val: 0.6107 acc_val: 0.8037 time: 0.2170s
Epoch: 0059 loss_train: 0.2366 acc_train: 0.9480 loss_val: 0.6315 acc_val: 0.7960 time: 0.2141s
Epoch: 0060 loss_train: 0.2416 acc_train: 0.9461 loss_val: 0.5899 acc_val: 0.8037 time: 0.2190s
Epoch: 0061 loss_train: 0.2258 acc_train: 0.9557 loss_val: 0.5607 acc_val: 0.8027 time: 0.2210s
Epoch: 0062 loss_train: 0.2169 acc_train: 0.9557 loss_val: 0.5824 acc_val: 0.7998 time: 0.2141s
Epoch: 0063 loss_train: 0.2161 acc_train: 0.9595 loss_val: 0.5798 acc_val: 0.8104 time: 0.2141s
Epoch: 0064 loss_train: 0.2229 acc_train: 0.9499 loss_val: 0.5725 acc_val: 0.8229 time: 0.2111s
Epoch: 0065 loss_train: 0.1982 acc_train: 0.9615 loss_val: 0.5417 acc_val: 0.8094 time: 0.2210s
Epoch: 0066 loss_train: 0.1954 acc_train: 0.9615 loss_val: 0.6399 acc_val: 0.8065 time: 0.2210s
Epoch: 0067 loss_train: 0.1939 acc_train: 0.9557 loss_val: 0.5562 acc_val: 0.8027 time: 0.2190s
Epoch: 0068 loss_train: 0.1739 acc_train: 0.9692 loss_val: 0.6468 acc_val: 0.8075 time: 0.2111s
Epoch: 0069 loss_train: 0.1742 acc_train: 0.9672 loss_val: 0.5700 acc_val: 0.8219 time: 0.2180s
Epoch: 0070 loss_train: 0.1677 acc_train: 0.9711 loss_val: 0.5423 acc_val: 0.8046 time: 0.2141s
Epoch: 0071 loss_train: 0.1669 acc_train: 0.9750 loss_val: 0.5493 acc_val: 0.8277 time: 0.2160s
Epoch: 0072 loss_train: 0.1557 acc_train: 0.9730 loss_val: 0.5190 acc_val: 0.8248 time: 0.2170s
Epoch: 0073 loss_train: 0.1562 acc_train: 0.9769 loss_val: 0.5456 acc_val: 0.8316 time: 0.2160s
Epoch: 0074 loss_train: 0.1486 acc_train: 0.9672 loss_val: 0.5618 acc_val: 0.8219 time: 0.2131s
Epoch: 0075 loss_train: 0.1401 acc_train: 0.9750 loss_val: 0.6169 acc_val: 0.8114 time: 0.2121s
Epoch: 0076 loss_train: 0.1427 acc_train: 0.9730 loss_val: 0.5355 acc_val: 0.8364 time: 0.2141s
Epoch: 0077 loss_train: 0.1310 acc_train: 0.9827 loss_val: 0.5949 acc_val: 0.8287 time: 0.2160s
Epoch: 0078 loss_train: 0.1309 acc_train: 0.9692 loss_val: 0.5119 acc_val: 0.8325 time: 0.2220s
Epoch: 0079 loss_train: 0.1326 acc_train: 0.9788 loss_val: 0.4569 acc_val: 0.8489 time: 0.2200s
Epoch: 0080 loss_train: 0.1080 acc_train: 0.9827 loss_val: 0.5041 acc_val: 0.8450 time: 0.2141s
Epoch: 0081 loss_train: 0.1009 acc_train: 0.9884 loss_val: 0.5197 acc_val: 0.8296 time: 0.2270s
Epoch: 0082 loss_train: 0.1125 acc_train: 0.9807 loss_val: 0.5485 acc_val: 0.8268 time: 0.2280s
Epoch: 0083 loss_train: 0.1084 acc_train: 0.9769 loss_val: 0.5429 acc_val: 0.8354 time: 0.2340s
Epoch: 0084 loss_train: 0.1062 acc_train: 0.9788 loss_val: 0.5328 acc_val: 0.8345 time: 0.2190s
Epoch: 0085 loss_train: 0.1040 acc_train: 0.9904 loss_val: 0.4852 acc_val: 0.8460 time: 0.2170s
Epoch: 0086 loss_train: 0.1226 acc_train: 0.9846 loss_val: 0.4992 acc_val: 0.8316 time: 0.2300s
Epoch: 0087 loss_train: 0.0883 acc_train: 0.9904 loss_val: 0.5762 acc_val: 0.8296 time: 0.2230s
Epoch: 0088 loss_train: 0.1036 acc_train: 0.9807 loss_val: 0.4998 acc_val: 0.8412 time: 0.2220s
Epoch: 0089 loss_train: 0.1127 acc_train: 0.9865 loss_val: 0.4945 acc_val: 0.8412 time: 0.2300s
Epoch: 0090 loss_train: 0.0932 acc_train: 0.9846 loss_val: 0.4652 acc_val: 0.8441 time: 0.2111s
Epoch: 0091 loss_train: 0.0967 acc_train: 0.9846 loss_val: 0.5492 acc_val: 0.8287 time: 0.2131s
Epoch: 0092 loss_train: 0.1096 acc_train: 0.9865 loss_val: 0.5112 acc_val: 0.8345 time: 0.2170s
Epoch: 0093 loss_train: 0.1018 acc_train: 0.9846 loss_val: 0.5066 acc_val: 0.8508 time: 0.2160s
Epoch: 0094 loss_train: 0.0886 acc_train: 0.9923 loss_val: 0.4691 acc_val: 0.8422 time: 0.2151s
Epoch: 0095 loss_train: 0.1059 acc_train: 0.9788 loss_val: 0.4833 acc_val: 0.8479 time: 0.2170s
Epoch: 0096 loss_train: 0.0892 acc_train: 0.9923 loss_val: 0.4524 acc_val: 0.8412 time: 0.2230s
Epoch: 0097 loss_train: 0.0948 acc_train: 0.9807 loss_val: 0.4591 acc_val: 0.8383 time: 0.2160s
Epoch: 0098 loss_train: 0.0754 acc_train: 0.9923 loss_val: 0.5741 acc_val: 0.8470 time: 0.2151s
Epoch: 0099 loss_train: 0.0767 acc_train: 0.9884 loss_val: 0.5295 acc_val: 0.8373 time: 0.2230s
Epoch: 0100 loss_train: 0.0856 acc_train: 0.9846 loss_val: 0.5241 acc_val: 0.8518 time: 0.2141s
Epoch: 0101 loss_train: 0.0646 acc_train: 0.9923 loss_val: 0.4928 acc_val: 0.8576 time: 0.2151s
Epoch: 0102 loss_train: 0.0820 acc_train: 0.9865 loss_val: 0.4610 acc_val: 0.8585 time: 0.2111s
Epoch: 0103 loss_train: 0.0803 acc_train: 0.9904 loss_val: 0.4798 acc_val: 0.8470 time: 0.2151s
Epoch: 0104 loss_train: 0.0788 acc_train: 0.9923 loss_val: 0.4548 acc_val: 0.8566 time: 0.2230s
Epoch: 0105 loss_train: 0.0796 acc_train: 0.9942 loss_val: 0.5646 acc_val: 0.8364 time: 0.2190s
Epoch: 0106 loss_train: 0.0841 acc_train: 0.9904 loss_val: 0.4878 acc_val: 0.8402 time: 0.2141s
Epoch: 0107 loss_train: 0.0845 acc_train: 0.9750 loss_val: 0.5314 acc_val: 0.8325 time: 0.2121s
Epoch: 0108 loss_train: 0.0740 acc_train: 0.9865 loss_val: 0.4464 acc_val: 0.8499 time: 0.2240s
Epoch: 0109 loss_train: 0.0801 acc_train: 0.9923 loss_val: 0.4886 acc_val: 0.8508 time: 0.2190s
Epoch: 0110 loss_train: 0.0675 acc_train: 0.9961 loss_val: 0.4860 acc_val: 0.8402 time: 0.2270s
Epoch: 0111 loss_train: 0.0739 acc_train: 0.9942 loss_val: 0.5559 acc_val: 0.8566 time: 0.2170s
Epoch: 0112 loss_train: 0.0572 acc_train: 0.9961 loss_val: 0.4987 acc_val: 0.8460 time: 0.2270s
Epoch: 0113 loss_train: 0.0667 acc_train: 0.9923 loss_val: 0.4824 acc_val: 0.8450 time: 0.2340s
Epoch: 0114 loss_train: 0.0733 acc_train: 0.9865 loss_val: 0.4975 acc_val: 0.8499 time: 0.2200s
Epoch: 0115 loss_train: 0.0689 acc_train: 0.9904 loss_val: 0.4946 acc_val: 0.8499 time: 0.2151s
Epoch: 0116 loss_train: 0.0754 acc_train: 0.9846 loss_val: 0.4877 acc_val: 0.8470 time: 0.2190s
Epoch: 0117 loss_train: 0.0723 acc_train: 0.9846 loss_val: 0.4627 acc_val: 0.8489 time: 0.2151s
Epoch: 0118 loss_train: 0.0667 acc_train: 0.9904 loss_val: 0.4579 acc_val: 0.8527 time: 0.2131s
Epoch: 0119 loss_train: 0.0782 acc_train: 0.9846 loss_val: 0.4782 acc_val: 0.8518 time: 0.2121s
Epoch: 0120 loss_train: 0.0666 acc_train: 0.9884 loss_val: 0.4929 acc_val: 0.8431 time: 0.2121s
Epoch: 0121 loss_train: 0.0787 acc_train: 0.9904 loss_val: 0.4728 acc_val: 0.8566 time: 0.2141s
Epoch: 0122 loss_train: 0.0617 acc_train: 0.9942 loss_val: 0.4788 acc_val: 0.8614 time: 0.2160s
Epoch: 0123 loss_train: 0.0633 acc_train: 0.9923 loss_val: 0.4512 acc_val: 0.8624 time: 0.2151s
Epoch: 0124 loss_train: 0.0713 acc_train: 0.9904 loss_val: 0.4956 acc_val: 0.8518 time: 0.2160s
Epoch: 0125 loss_train: 0.0879 acc_train: 0.9788 loss_val: 0.4857 acc_val: 0.8518 time: 0.2121s
Epoch: 0126 loss_train: 0.0693 acc_train: 0.9846 loss_val: 0.4731 acc_val: 0.8508 time: 0.2250s
Epoch: 0127 loss_train: 0.0812 acc_train: 0.9827 loss_val: 0.4966 acc_val: 0.8470 time: 0.2380s
Epoch: 0128 loss_train: 0.0737 acc_train: 0.9923 loss_val: 0.4925 acc_val: 0.8422 time: 0.2190s
Epoch: 0129 loss_train: 0.0627 acc_train: 0.9961 loss_val: 0.4564 acc_val: 0.8614 time: 0.2141s
Epoch: 0130 loss_train: 0.0721 acc_train: 0.9904 loss_val: 0.4791 acc_val: 0.8422 time: 0.2190s
Epoch: 0131 loss_train: 0.0721 acc_train: 0.9923 loss_val: 0.4575 acc_val: 0.8566 time: 0.2151s
Epoch: 0132 loss_train: 0.0641 acc_train: 0.9981 loss_val: 0.4704 acc_val: 0.8604 time: 0.2170s
Epoch: 0133 loss_train: 0.0664 acc_train: 0.9923 loss_val: 0.4843 acc_val: 0.8499 time: 0.2121s
Epoch: 0134 loss_train: 0.0626 acc_train: 0.9942 loss_val: 0.4978 acc_val: 0.8499 time: 0.2290s
Epoch: 0135 loss_train: 0.0720 acc_train: 0.9923 loss_val: 0.5023 acc_val: 0.8614 time: 0.2300s
Epoch: 0136 loss_train: 0.0701 acc_train: 0.9865 loss_val: 0.5441 acc_val: 0.8431 time: 0.2131s
Epoch: 0137 loss_train: 0.0714 acc_train: 0.9865 loss_val: 0.4743 acc_val: 0.8431 time: 0.2131s
Epoch: 0138 loss_train: 0.0647 acc_train: 0.9942 loss_val: 0.4156 acc_val: 0.8691 time: 0.2210s
Epoch: 0139 loss_train: 0.0702 acc_train: 0.9923 loss_val: 0.4672 acc_val: 0.8768 time: 0.2151s
Epoch: 0140 loss_train: 0.0700 acc_train: 0.9846 loss_val: 0.4525 acc_val: 0.8576 time: 0.2121s
Epoch: 0141 loss_train: 0.0629 acc_train: 0.9923 loss_val: 0.4998 acc_val: 0.8441 time: 0.2170s
Epoch: 0142 loss_train: 0.0602 acc_train: 0.9981 loss_val: 0.5014 acc_val: 0.8441 time: 0.2111s
Epoch: 0143 loss_train: 0.0706 acc_train: 0.9865 loss_val: 0.4279 acc_val: 0.8566 time: 0.2121s
Epoch: 0144 loss_train: 0.0676 acc_train: 0.9923 loss_val: 0.4758 acc_val: 0.8614 time: 0.2200s
Epoch: 0145 loss_train: 0.0678 acc_train: 0.9904 loss_val: 0.5081 acc_val: 0.8576 time: 0.2160s
Epoch: 0146 loss_train: 0.0649 acc_train: 0.9904 loss_val: 0.4565 acc_val: 0.8499 time: 0.2160s
Epoch: 0147 loss_train: 0.0843 acc_train: 0.9827 loss_val: 0.4702 acc_val: 0.8508 time: 0.2121s
Epoch: 0148 loss_train: 0.0637 acc_train: 0.9884 loss_val: 0.5184 acc_val: 0.8441 time: 0.2131s
Epoch: 0149 loss_train: 0.0719 acc_train: 0.9884 loss_val: 0.5414 acc_val: 0.8345 time: 0.2160s
Epoch: 0150 loss_train: 0.0595 acc_train: 0.9981 loss_val: 0.4740 acc_val: 0.8373 time: 0.2210s
Epoch: 0151 loss_train: 0.0583 acc_train: 0.9904 loss_val: 0.4189 acc_val: 0.8653 time: 0.2170s
Epoch: 0152 loss_train: 0.0584 acc_train: 0.9923 loss_val: 0.4413 acc_val: 0.8547 time: 0.2160s
Epoch: 0153 loss_train: 0.0604 acc_train: 0.9904 loss_val: 0.4793 acc_val: 0.8508 time: 0.2121s
Epoch: 0154 loss_train: 0.0624 acc_train: 0.9904 loss_val: 0.4481 acc_val: 0.8489 time: 0.2210s
Epoch: 0155 loss_train: 0.0491 acc_train: 0.9942 loss_val: 0.4600 acc_val: 0.8585 time: 0.2141s
Epoch: 0156 loss_train: 0.0653 acc_train: 0.9923 loss_val: 0.5274 acc_val: 0.8345 time: 0.2170s
Epoch: 0157 loss_train: 0.0628 acc_train: 0.9904 loss_val: 0.4982 acc_val: 0.8431 time: 0.2131s
Epoch: 0158 loss_train: 0.0564 acc_train: 0.9923 loss_val: 0.4928 acc_val: 0.8460 time: 0.2151s
Epoch: 0159 loss_train: 0.0575 acc_train: 0.9942 loss_val: 0.4169 acc_val: 0.8566 time: 0.2230s
Epoch: 0160 loss_train: 0.0617 acc_train: 0.9884 loss_val: 0.4782 acc_val: 0.8662 time: 0.2141s
Epoch: 0161 loss_train: 0.0516 acc_train: 0.9961 loss_val: 0.4841 acc_val: 0.8537 time: 0.2170s
Epoch: 0162 loss_train: 0.0491 acc_train: 0.9981 loss_val: 0.4972 acc_val: 0.8633 time: 0.2160s
Epoch: 0163 loss_train: 0.0586 acc_train: 0.9904 loss_val: 0.4998 acc_val: 0.8393 time: 0.2240s
Epoch: 0164 loss_train: 0.0529 acc_train: 0.9923 loss_val: 0.4652 acc_val: 0.8566 time: 0.2160s
Epoch: 0165 loss_train: 0.0614 acc_train: 0.9961 loss_val: 0.4788 acc_val: 0.8537 time: 0.2111s
Epoch: 0166 loss_train: 0.0736 acc_train: 0.9807 loss_val: 0.4605 acc_val: 0.8547 time: 0.2230s
Epoch: 0167 loss_train: 0.0580 acc_train: 0.9904 loss_val: 0.4760 acc_val: 0.8422 time: 0.2200s
Epoch: 0168 loss_train: 0.0487 acc_train: 0.9942 loss_val: 0.4237 acc_val: 0.8537 time: 0.2151s
Epoch: 0169 loss_train: 0.0526 acc_train: 0.9942 loss_val: 0.4739 acc_val: 0.8643 time: 0.2121s
Epoch: 0170 loss_train: 0.0543 acc_train: 0.9923 loss_val: 0.4258 acc_val: 0.8681 time: 0.2210s
Epoch: 0171 loss_train: 0.0697 acc_train: 0.9884 loss_val: 0.4868 acc_val: 0.8508 time: 0.2160s
Epoch: 0172 loss_train: 0.0612 acc_train: 0.9923 loss_val: 0.4808 acc_val: 0.8518 time: 0.2190s
Epoch: 0173 loss_train: 0.0561 acc_train: 0.9904 loss_val: 0.4744 acc_val: 0.8576 time: 0.2151s
Epoch: 0174 loss_train: 0.0692 acc_train: 0.9904 loss_val: 0.4775 acc_val: 0.8508 time: 0.2141s
Epoch: 0175 loss_train: 0.0502 acc_train: 0.9942 loss_val: 0.5157 acc_val: 0.8499 time: 0.2131s
Epoch: 0176 loss_train: 0.0571 acc_train: 0.9961 loss_val: 0.4509 acc_val: 0.8537 time: 0.2160s
Epoch: 0177 loss_train: 0.0655 acc_train: 0.9904 loss_val: 0.4463 acc_val: 0.8499 time: 0.2160s
Epoch: 0178 loss_train: 0.0568 acc_train: 0.9961 loss_val: 0.5100 acc_val: 0.8422 time: 0.2180s
Epoch: 0179 loss_train: 0.0582 acc_train: 0.9942 loss_val: 0.4484 acc_val: 0.8499 time: 0.2170s
Epoch: 0180 loss_train: 0.0646 acc_train: 0.9923 loss_val: 0.4765 acc_val: 0.8576 time: 0.2200s
Epoch: 0181 loss_train: 0.0534 acc_train: 0.9942 loss_val: 0.5141 acc_val: 0.8412 time: 0.2230s
Epoch: 0182 loss_train: 0.0517 acc_train: 0.9904 loss_val: 0.5056 acc_val: 0.8373 time: 0.2121s
Epoch: 0183 loss_train: 0.0546 acc_train: 0.9961 loss_val: 0.4999 acc_val: 0.8537 time: 0.2160s
Epoch: 0184 loss_train: 0.0506 acc_train: 0.9942 loss_val: 0.4662 acc_val: 0.8556 time: 0.2141s
Epoch: 0185 loss_train: 0.0484 acc_train: 0.9942 loss_val: 0.4924 acc_val: 0.8422 time: 0.2131s
Epoch: 0186 loss_train: 0.0481 acc_train: 0.9923 loss_val: 0.4615 acc_val: 0.8527 time: 0.2180s
Epoch: 0187 loss_train: 0.0535 acc_train: 0.9942 loss_val: 0.4659 acc_val: 0.8547 time: 0.2141s
Epoch: 0188 loss_train: 0.0500 acc_train: 0.9923 loss_val: 0.4546 acc_val: 0.8604 time: 0.2190s
Epoch: 0189 loss_train: 0.0444 acc_train: 0.9961 loss_val: 0.4539 acc_val: 0.8537 time: 0.2141s
Epoch: 0190 loss_train: 0.0663 acc_train: 0.9865 loss_val: 0.4705 acc_val: 0.8527 time: 0.2200s
Epoch: 0191 loss_train: 0.0536 acc_train: 0.9904 loss_val: 0.4501 acc_val: 0.8566 time: 0.2220s
Epoch: 0192 loss_train: 0.0540 acc_train: 0.9942 loss_val: 0.4704 acc_val: 0.8508 time: 0.2220s
Epoch: 0193 loss_train: 0.0530 acc_train: 0.9961 loss_val: 0.5044 acc_val: 0.8402 time: 0.2160s
Epoch: 0194 loss_train: 0.0452 acc_train: 0.9942 loss_val: 0.4221 acc_val: 0.8681 time: 0.2160s
Epoch: 0195 loss_train: 0.0549 acc_train: 0.9884 loss_val: 0.4717 acc_val: 0.8576 time: 0.2160s
Epoch: 0196 loss_train: 0.0454 acc_train: 0.9942 loss_val: 0.4634 acc_val: 0.8422 time: 0.2141s
Epoch: 0197 loss_train: 0.0536 acc_train: 0.9884 loss_val: 0.4712 acc_val: 0.8499 time: 0.2151s
Epoch: 0198 loss_train: 0.0505 acc_train: 0.9961 loss_val: 0.4753 acc_val: 0.8624 time: 0.2180s
Epoch: 0199 loss_train: 0.0605 acc_train: 0.9827 loss_val: 0.4500 acc_val: 0.8585 time: 0.2131s
Epoch: 0200 loss_train: 0.0555 acc_train: 0.9961 loss_val: 0.4760 acc_val: 0.8470 time: 0.2160s
Epoch: 0201 loss_train: 0.0460 acc_train: 0.9981 loss_val: 0.4694 acc_val: 0.8547 time: 0.2151s
Epoch: 0202 loss_train: 0.0522 acc_train: 0.9923 loss_val: 0.4909 acc_val: 0.8681 time: 0.2151s
Epoch: 0203 loss_train: 0.0469 acc_train: 0.9981 loss_val: 0.4877 acc_val: 0.8527 time: 0.2220s
Epoch: 0204 loss_train: 0.0512 acc_train: 0.9961 loss_val: 0.4654 acc_val: 0.8537 time: 0.2190s
Epoch: 0205 loss_train: 0.0385 acc_train: 0.9961 loss_val: 0.4967 acc_val: 0.8489 time: 0.2160s
Epoch: 0206 loss_train: 0.0519 acc_train: 0.9942 loss_val: 0.4935 acc_val: 0.8479 time: 0.2250s
Epoch: 0207 loss_train: 0.0434 acc_train: 0.9942 loss_val: 0.4530 acc_val: 0.8595 time: 0.2330s
Epoch: 0208 loss_train: 0.0499 acc_train: 0.9942 loss_val: 0.4885 acc_val: 0.8556 time: 0.2170s
Epoch: 0209 loss_train: 0.0483 acc_train: 0.9923 loss_val: 0.4791 acc_val: 0.8460 time: 0.2160s
Epoch: 0210 loss_train: 0.0479 acc_train: 0.9942 loss_val: 0.4605 acc_val: 0.8527 time: 0.2240s
Epoch: 0211 loss_train: 0.0555 acc_train: 0.9865 loss_val: 0.4509 acc_val: 0.8614 time: 0.2121s
Epoch: 0212 loss_train: 0.0525 acc_train: 0.9942 loss_val: 0.4755 acc_val: 0.8499 time: 0.2190s
Epoch: 0213 loss_train: 0.0521 acc_train: 0.9923 loss_val: 0.4353 acc_val: 0.8537 time: 0.2190s
Epoch: 0214 loss_train: 0.0428 acc_train: 0.9923 loss_val: 0.4652 acc_val: 0.8566 time: 0.2131s
Epoch: 0215 loss_train: 0.0495 acc_train: 0.9942 loss_val: 0.4364 acc_val: 0.8643 time: 0.2160s
Epoch: 0216 loss_train: 0.0396 acc_train: 0.9981 loss_val: 0.4796 acc_val: 0.8547 time: 0.2131s
Epoch: 0217 loss_train: 0.0389 acc_train: 0.9961 loss_val: 0.4595 acc_val: 0.8653 time: 0.2240s
Epoch: 0218 loss_train: 0.0570 acc_train: 0.9846 loss_val: 0.4408 acc_val: 0.8633 time: 0.2151s
Epoch: 0219 loss_train: 0.0494 acc_train: 0.9923 loss_val: 0.4357 acc_val: 0.8691 time: 0.2141s
Epoch: 0220 loss_train: 0.0499 acc_train: 0.9923 loss_val: 0.4586 acc_val: 0.8556 time: 0.2260s
Epoch: 0221 loss_train: 0.0507 acc_train: 0.9961 loss_val: 0.4781 acc_val: 0.8556 time: 0.2141s
Epoch: 0222 loss_train: 0.0489 acc_train: 0.9904 loss_val: 0.5225 acc_val: 0.8508 time: 0.2190s
Epoch: 0223 loss_train: 0.0495 acc_train: 0.9961 loss_val: 0.4594 acc_val: 0.8547 time: 0.2160s
Epoch: 0224 loss_train: 0.0474 acc_train: 0.9961 loss_val: 0.4977 acc_val: 0.8499 time: 0.2160s
Epoch: 0225 loss_train: 0.0434 acc_train: 0.9942 loss_val: 0.4290 acc_val: 0.8614 time: 0.2220s
Epoch: 0226 loss_train: 0.0386 acc_train: 1.0000 loss_val: 0.5077 acc_val: 0.8441 time: 0.2141s
Epoch: 0227 loss_train: 0.0452 acc_train: 0.9923 loss_val: 0.4769 acc_val: 0.8489 time: 0.2180s
Epoch: 0228 loss_train: 0.0480 acc_train: 0.9942 loss_val: 0.4695 acc_val: 0.8537 time: 0.2131s
Epoch: 0229 loss_train: 0.0531 acc_train: 0.9981 loss_val: 0.4753 acc_val: 0.8547 time: 0.2200s
Epoch: 0230 loss_train: 0.0450 acc_train: 0.9923 loss_val: 0.4863 acc_val: 0.8527 time: 0.2290s
Epoch: 0231 loss_train: 0.0497 acc_train: 0.9942 loss_val: 0.4882 acc_val: 0.8489 time: 0.2170s
Epoch: 0232 loss_train: 0.0368 acc_train: 0.9961 loss_val: 0.4368 acc_val: 0.8633 time: 0.2270s
Epoch: 0233 loss_train: 0.0495 acc_train: 0.9942 loss_val: 0.4588 acc_val: 0.8653 time: 0.2141s
Epoch: 0234 loss_train: 0.0431 acc_train: 0.9981 loss_val: 0.4830 acc_val: 0.8489 time: 0.2170s
Epoch: 0235 loss_train: 0.0415 acc_train: 0.9923 loss_val: 0.5391 acc_val: 0.8489 time: 0.2160s
Epoch: 0236 loss_train: 0.0436 acc_train: 0.9923 loss_val: 0.5022 acc_val: 0.8489 time: 0.2160s
Epoch: 0237 loss_train: 0.0450 acc_train: 0.9942 loss_val: 0.5367 acc_val: 0.8489 time: 0.2141s
Epoch: 0238 loss_train: 0.0446 acc_train: 0.9961 loss_val: 0.4710 acc_val: 0.8518 time: 0.2180s
Epoch: 0239 loss_train: 0.0373 acc_train: 0.9942 loss_val: 0.5111 acc_val: 0.8556 time: 0.2250s
Epoch: 0240 loss_train: 0.0399 acc_train: 0.9981 loss_val: 0.5125 acc_val: 0.8576 time: 0.2170s
Epoch: 0241 loss_train: 0.0364 acc_train: 0.9961 loss_val: 0.4955 acc_val: 0.8479 time: 0.2210s
Epoch: 0242 loss_train: 0.0507 acc_train: 0.9846 loss_val: 0.5619 acc_val: 0.8489 time: 0.2151s
Epoch: 0243 loss_train: 0.0369 acc_train: 0.9961 loss_val: 0.5317 acc_val: 0.8537 time: 0.2131s
Epoch: 0244 loss_train: 0.0404 acc_train: 1.0000 loss_val: 0.5770 acc_val: 0.8441 time: 0.2141s
Epoch: 0245 loss_train: 0.0577 acc_train: 0.9884 loss_val: 0.4955 acc_val: 0.8422 time: 0.2160s
Epoch: 0246 loss_train: 0.0531 acc_train: 0.9942 loss_val: 0.4422 acc_val: 0.8624 time: 0.2141s
Epoch: 0247 loss_train: 0.0444 acc_train: 0.9981 loss_val: 0.4607 acc_val: 0.8547 time: 0.2160s
Epoch: 0248 loss_train: 0.0511 acc_train: 0.9942 loss_val: 0.4384 acc_val: 0.8672 time: 0.2180s
Epoch: 0249 loss_train: 0.0468 acc_train: 0.9942 loss_val: 0.4432 acc_val: 0.8662 time: 0.2210s
Epoch: 0250 loss_train: 0.0375 acc_train: 0.9981 loss_val: 0.4852 acc_val: 0.8576 time: 0.2210s
Epoch: 0251 loss_train: 0.0484 acc_train: 0.9923 loss_val: 0.4753 acc_val: 0.8576 time: 0.2330s
Epoch: 0252 loss_train: 0.0494 acc_train: 0.9942 loss_val: 0.5026 acc_val: 0.8431 time: 0.2121s
Epoch: 0253 loss_train: 0.0513 acc_train: 0.9884 loss_val: 0.4861 acc_val: 0.8499 time: 0.2131s
Epoch: 0254 loss_train: 0.0439 acc_train: 0.9942 loss_val: 0.4705 acc_val: 0.8508 time: 0.2131s
Epoch: 0255 loss_train: 0.0433 acc_train: 0.9923 loss_val: 0.4554 acc_val: 0.8585 time: 0.2210s
Epoch: 0256 loss_train: 0.0389 acc_train: 0.9961 loss_val: 0.4673 acc_val: 0.8527 time: 0.2170s
Epoch: 0257 loss_train: 0.0481 acc_train: 0.9961 loss_val: 0.4485 acc_val: 0.8614 time: 0.2170s
Epoch: 0258 loss_train: 0.0299 acc_train: 0.9981 loss_val: 0.5305 acc_val: 0.8460 time: 0.2131s
Epoch: 0259 loss_train: 0.0460 acc_train: 0.9923 loss_val: 0.4666 acc_val: 0.8527 time: 0.2180s
Epoch: 0260 loss_train: 0.0449 acc_train: 0.9923 loss_val: 0.4732 acc_val: 0.8537 time: 0.2210s
Epoch: 0261 loss_train: 0.0420 acc_train: 0.9961 loss_val: 0.4752 acc_val: 0.8479 time: 0.2220s
Epoch: 0262 loss_train: 0.0423 acc_train: 0.9961 loss_val: 0.4841 acc_val: 0.8402 time: 0.2141s
Epoch: 0263 loss_train: 0.0362 acc_train: 0.9961 loss_val: 0.4992 acc_val: 0.8441 time: 0.2270s
Epoch: 0264 loss_train: 0.0387 acc_train: 0.9923 loss_val: 0.4966 acc_val: 0.8585 time: 0.2230s
Epoch: 0265 loss_train: 0.0390 acc_train: 0.9923 loss_val: 0.4611 acc_val: 0.8441 time: 0.2220s
Epoch: 0266 loss_train: 0.0344 acc_train: 0.9981 loss_val: 0.5262 acc_val: 0.8508 time: 0.2170s
Epoch: 0267 loss_train: 0.0415 acc_train: 0.9981 loss_val: 0.5052 acc_val: 0.8576 time: 0.2170s
Epoch: 0268 loss_train: 0.0426 acc_train: 0.9942 loss_val: 0.5075 acc_val: 0.8499 time: 0.2180s
Epoch: 0269 loss_train: 0.0426 acc_train: 0.9961 loss_val: 0.4320 acc_val: 0.8499 time: 0.2131s
Epoch: 0270 loss_train: 0.0433 acc_train: 0.9961 loss_val: 0.4778 acc_val: 0.8624 time: 0.2141s
Epoch: 0271 loss_train: 0.0447 acc_train: 0.9942 loss_val: 0.4520 acc_val: 0.8556 time: 0.2180s
Epoch: 0272 loss_train: 0.0423 acc_train: 0.9923 loss_val: 0.4792 acc_val: 0.8518 time: 0.2131s
Epoch: 0273 loss_train: 0.0443 acc_train: 0.9884 loss_val: 0.4429 acc_val: 0.8479 time: 0.2210s
Epoch: 0274 loss_train: 0.0417 acc_train: 0.9961 loss_val: 0.4357 acc_val: 0.8653 time: 0.2250s
Epoch: 0275 loss_train: 0.0379 acc_train: 1.0000 loss_val: 0.4643 acc_val: 0.8585 time: 0.2210s
Epoch: 0276 loss_train: 0.0329 acc_train: 0.9981 loss_val: 0.4609 acc_val: 0.8585 time: 0.2220s
Epoch: 0277 loss_train: 0.0377 acc_train: 0.9961 loss_val: 0.4725 acc_val: 0.8585 time: 0.2200s
Epoch: 0278 loss_train: 0.0387 acc_train: 0.9981 loss_val: 0.4773 acc_val: 0.8537 time: 0.2180s
Epoch: 0279 loss_train: 0.0414 acc_train: 0.9942 loss_val: 0.4600 acc_val: 0.8547 time: 0.2170s
Epoch: 0280 loss_train: 0.0401 acc_train: 0.9961 loss_val: 0.4882 acc_val: 0.8422 time: 0.2200s
Epoch: 0281 loss_train: 0.0409 acc_train: 0.9942 loss_val: 0.4777 acc_val: 0.8460 time: 0.2170s
Epoch: 0282 loss_train: 0.0364 acc_train: 0.9961 loss_val: 0.4986 acc_val: 0.8537 time: 0.2200s
Epoch: 0283 loss_train: 0.0418 acc_train: 0.9981 loss_val: 0.4835 acc_val: 0.8527 time: 0.2151s
Epoch: 0284 loss_train: 0.0258 acc_train: 1.0000 loss_val: 0.4725 acc_val: 0.8527 time: 0.2360s
Epoch: 0285 loss_train: 0.0331 acc_train: 0.9961 loss_val: 0.4677 acc_val: 0.8470 time: 0.2180s
Epoch: 0286 loss_train: 0.0449 acc_train: 0.9942 loss_val: 0.4734 acc_val: 0.8460 time: 0.2141s
Epoch: 0287 loss_train: 0.0372 acc_train: 1.0000 loss_val: 0.4826 acc_val: 0.8624 time: 0.2160s
Epoch: 0288 loss_train: 0.0330 acc_train: 0.9961 loss_val: 0.4896 acc_val: 0.8499 time: 0.2180s
Epoch: 0289 loss_train: 0.0484 acc_train: 0.9904 loss_val: 0.5417 acc_val: 0.8614 time: 0.2121s
Epoch: 0290 loss_train: 0.0322 acc_train: 0.9981 loss_val: 0.5361 acc_val: 0.8412 time: 0.2180s
Epoch: 0291 loss_train: 0.0407 acc_train: 0.9904 loss_val: 0.5178 acc_val: 0.8489 time: 0.2180s
Epoch: 0292 loss_train: 0.0349 acc_train: 0.9923 loss_val: 0.4534 acc_val: 0.8470 time: 0.2250s
Epoch: 0293 loss_train: 0.0541 acc_train: 0.9884 loss_val: 0.4883 acc_val: 0.8556 time: 0.2170s
Epoch: 0294 loss_train: 0.0340 acc_train: 0.9961 loss_val: 0.4723 acc_val: 0.8672 time: 0.2210s
Epoch: 0295 loss_train: 0.0373 acc_train: 0.9961 loss_val: 0.4664 acc_val: 0.8489 time: 0.2200s
Epoch: 0296 loss_train: 0.0352 acc_train: 0.9981 loss_val: 0.4232 acc_val: 0.8710 time: 0.2160s
Epoch: 0297 loss_train: 0.0407 acc_train: 0.9981 loss_val: 0.4645 acc_val: 0.8643 time: 0.2220s
Epoch: 0298 loss_train: 0.0399 acc_train: 0.9961 loss_val: 0.4564 acc_val: 0.8547 time: 0.2160s
Epoch: 0299 loss_train: 0.0379 acc_train: 0.9981 loss_val: 0.5097 acc_val: 0.8556 time: 0.2160s
Epoch: 0300 loss_train: 0.0404 acc_train: 0.9942 loss_val: 0.4524 acc_val: 0.8595 time: 0.2200s
Pretraining process finished ! 
Epoch: 0001 loss_train: 0.0819 acc_train: 0.9865 loss_val: 0.3237 acc_val: 0.8912 time: 0.4162s
Epoch: 0002 loss_train: 0.0764 acc_train: 0.9884 loss_val: 0.3406 acc_val: 0.8912 time: 0.4152s
Epoch: 0003 loss_train: 0.0798 acc_train: 0.9923 loss_val: 0.3544 acc_val: 0.8864 time: 0.4102s
Epoch: 0004 loss_train: 0.0689 acc_train: 0.9923 loss_val: 0.3573 acc_val: 0.8874 time: 0.4082s
Epoch: 0005 loss_train: 0.0617 acc_train: 0.9904 loss_val: 0.3507 acc_val: 0.8874 time: 0.4251s
Epoch: 0006 loss_train: 0.0644 acc_train: 0.9904 loss_val: 0.3408 acc_val: 0.8864 time: 0.4102s
Epoch: 0007 loss_train: 0.0663 acc_train: 0.9807 loss_val: 0.3317 acc_val: 0.8884 time: 0.4062s
Epoch: 0008 loss_train: 0.0521 acc_train: 0.9961 loss_val: 0.3240 acc_val: 0.8941 time: 0.4092s
Epoch: 0009 loss_train: 0.0650 acc_train: 0.9807 loss_val: 0.3177 acc_val: 0.8999 time: 0.4122s
Epoch: 0010 loss_train: 0.0632 acc_train: 0.9923 loss_val: 0.3146 acc_val: 0.8989 time: 0.4112s
Epoch: 0011 loss_train: 0.0555 acc_train: 0.9942 loss_val: 0.3128 acc_val: 0.8999 time: 0.4152s
Epoch: 0012 loss_train: 0.0500 acc_train: 0.9961 loss_val: 0.3124 acc_val: 0.9009 time: 0.4162s
Epoch: 0013 loss_train: 0.0487 acc_train: 0.9942 loss_val: 0.3127 acc_val: 0.9009 time: 0.4221s
Epoch: 0014 loss_train: 0.0463 acc_train: 0.9961 loss_val: 0.3140 acc_val: 0.9009 time: 0.4152s
Epoch: 0015 loss_train: 0.0525 acc_train: 0.9942 loss_val: 0.3167 acc_val: 0.9009 time: 0.4211s
Epoch: 0016 loss_train: 0.0452 acc_train: 0.9923 loss_val: 0.3210 acc_val: 0.8980 time: 0.4132s
Epoch: 0017 loss_train: 0.0490 acc_train: 0.9942 loss_val: 0.3247 acc_val: 0.8961 time: 0.4251s
Epoch: 0018 loss_train: 0.0510 acc_train: 0.9904 loss_val: 0.3283 acc_val: 0.8961 time: 0.4092s
Early stop  ! 
Optimization Finished!
Total time elapsed: 7.4661s
Test set results: loss= 0.3265 accuracy= 0.8997

Process finished with exit code 0
Configure as you like:
```
