# Adviser Networks: Learning What Question to Ask for Human-In-The-Loop Viewpoint Estimation

By [Mohamed El Banani](http://mbanani.github.io/) and [Jason J. Corso](http://web.eecs.umich.edu/~jjcorso/), University of Michigan


### Introduction

This is the code and data for the work presented in the arXiv tech report, [Adviser Networks](https://arxiv.org/abs/1802.01666).

If you find this work useful in your research, please consider citing:

    @ARTICLE{elbanani_adviser_2018,
        author = "El Banani, Mohamed and Corso, Jason J.",
        title = "Adviser Networks: Learning What Question to Ask for Human-In-The-Loop Viewpoint Estimation",
        journal = ArXiv e-prints,
        year = 2018,
        month = March,
    }


If you have any questions, please email me at mbanani@umich.edu.


### To Do

All code uploaded. README will be updated soon.

<!-- ### To Do

- [x] Look at the tasks described by the ICLR 2018 paper -- ["Ask The Right Question"](https://openreview.net/forum?id=S1CChZ-CZ)
    - While the paper seems to be of relevance at the high level,
        the problem tackled is very different, and very specific to language QA
- [x] Better metrics -- include priors calculated properly!
    - Priors are currently calculated using the same dataset.
- [x] Move viewpoint estimation code to Adviser
    - [x] Figure out a nice way of including everything in the same repo ?!
    - [x] Remove dependancy on old weights (NPY and PTH)
    - [x] Output a saved dictionary that Adviser can easily operate over
- [x] Import adviser code from the viewpoint_estimation repository!
    - [x] Move Code!
    - [x] Edit dataset wrapper to be more comprehensible
    - [x] Replicate results on Caffe model estimates -- ignore!
    - [x] Run Adviser on the new attention-FT model
        - Results `python train_adviser.py --dataset advisee_full --model alexAdviser --temperature 0.01`
            - Accuracy  :  [93.95 89.03 84.07]  -- mean :  89.02
            - Geo Dist  :  [ 3.48  5.75 12.93]  -- mean :  7.39
    - [x] Implement saving!
- [ ] Implement different form of attentional stream
    - One idea is to apply a cross product to an augmented version of the KPC,
      convolve that, and then use that as the attentional map ?
     - [x] Implement idea
     - [x] Test idea -- reaches accuracy of 82%, which is higher than what is achieved by baseline CH-CNN, but lower than R4CNN-FT
     - [ ] Test reversal of KP-Map; the current method has 1s outside of locations and 0s near it .. which seems counter intuitive
- [ ] Implement on a different task -- Fine-grained classification
    - [x] Augment original dataset wrappers for the task
    - [ ] Test baseline alexnet methods on bird dataset
        - ~~test on bird snap~~ doesn't really work, got accuracy of nearly 12%
        - ~~test on CUB~~ doesn't really work, got accuracy of nearly 12%
    - [ ] Augment models to apply to new attention scheme
    - [ ] Augment AlexNet to operate over multiple branches
        - Use model pretrained on ImageNet
        - Zero-center and Normalize input to model
        - Use attention-based optimization and compare to normal optimization scheme.
- [ ] Compare against Bayesian, Information Theory, (~~and RL ?~~) approaches to this problem
- [ ] Move prior calculation outside of baseline calculation in adviser_metrics
- [ ] Move logging inside of metrics
- [ ] Add `with torch.no_grad():` to prevent gradient calculation for evaluation
- [ ] Fix error with `loss_weights`
- [ ] Move to tensorboardX to avoid internal dependancy on tensorflow

### Results

#### Pascal3D - Vehicles with Keypoints -- Overall Results

We fine-tuned both models on the Pascal 3D+ (Vehicles with Keypoints) dataset.
Since we suspect that the problem with the replication of the Click-Here CNN model
is in the attention section, we conducted an experiment where we only fine-tuned
those weights. As reported below, fine-tuning just the attention model achieves the best performance.

|                               |  bus  | car   | m.bike | mean  |  bus   | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|:------:|:-----:|:------:|:-----:|
| Render For CNN                | 89.26 | 74.36 | 81.93  | 81.85 |  5.16  | 8.53  | 13.46  | 9.05  |
| Render For CNN FT             | 93.55 | 83.98 | 87.30  | 88.28 |  3.04  | 5.83  | 11.95  | 6.94  |
| Render For CNN FT (reported)  | 90.6  | 82.4  | 84.1   | 85.7  |  2.93  | 5.63  | 11.7   | 6.74  |
| Click-Here CNN                | 86.91 | 83.25 | 73.83  | 81.33 |  4.01  | 8.18  | 19.71  | 10.63 |
| Click-Here CNN (reported)     | 96.8  | 90.2  | 85.2   | 90.7  |  2.63  | 4.98  | 11.4   | 6.35  |
| Click-Here CNN FT             | 92.97 | 89.84 | 81.25  | 88.02 |  2.93  | 5.14  | 13.42  | 7.16  |
| Click-Here CNN FT-Attention   | 94.48 | 90.77 | 84.91  | 90.05 |  2.88  | 5.24  | 12.10  | 6.74  |



#### Pascal KP for Adviser  -- Baselines

##### Clickhere CNN -- PyTorch Finetuned on Attention

| Train Set  |  bus  | car   | m.bike | mean  |  bus  | car   | m.bike | mean  |
|:----------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------:|:-----:|
| Worst      | 99.63 | 99.26 | 95.59  | 98.16 | 0.    | 0.71  | 0.     | 0.24  |
| Mean       | 100.  | 99.68 | 96.18  | 98.62 | 0.    | 0.12  | 0.     | 0.04  |
| Median     | 100.  | 99.79 | 96.18  | 98.66 | 0.    | 0.    | 0.     | 0.0   |
| Best       | 100.  | 99.89 | 96.47  | 98.79 | 0.    | 0.    | 0.     | 0.0   |


| Test Set   |  bus  | car   | m.bike | mean  |  bus  | car   | m.bike | mean  |
|:----------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------:|:-----:|
| Worst      | 87.19 | 81.4  | 75.81  | 81.47 |  3.7  | 7.1   | 13.98  | 8.26  |
| Mean       | 89.68 | 84.73 | 78.17  | 84.19 |  3.41 | 6.65  | 13.21  | 7.76  |
| Median     | 92.17 | 88.71 | 79.35  | 86.74 |  3.25 | 6.18  | 12.71  | 7.38  |
| Best       | 95.37 | 92.37 | 87.91  | 91.88 |  2.92 | 5.45  | 11.69  | 6.69  |
 -->


## Acknowledgements

This work has been partially supported by DARPA W32P4Q-15-C-0070 (subcontract from SoarTech) and funds from the University of Michigan Mobility Transformation Center.
