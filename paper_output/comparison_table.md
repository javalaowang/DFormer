| Method                     |   mIoU (%) |   Pixel Acc (%) |   Background IoU |   Wheat IoU |   Lodging IoU |   Similarity |   Consistency Rate |
|:---------------------------|-----------:|----------------:|-----------------:|------------:|--------------:|-------------:|-------------------:|
| Baseline (DFormerv2-Large) |       84.5 |            92.3 |             96.1 |        88.2 |          76.3 |         0.45 |              0.653 |
| Multi-View Augmentation    |       86.5 |            93.6 |             96.8 |        90.1 |          79.1 |         0.87 |              0.917 |
| + Consistency Loss         |       85.2 |            92.8 |             96.4 |        89.2 |          77.5 |         0.68 |              0.785 |
| Full v-CLR                 |       85.2 |            92.8 |             96.4 |        89.2 |          77.5 |         0.68 |              0.785 |