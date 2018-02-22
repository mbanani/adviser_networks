# Adviser Networks: Learning What Question to Ask for Human-In-The-Loop Viewpoint Estimation

By [Mohamed El Banani](http://mbanani.github.io/) and [Jason J. Corso](http://web.eecs.umich.edu/~jjcorso/), University of Michigan


## Introduction

This is the code and data for the work presented in the arXiv tech report, [Adviser Networks](https://arxiv.org/abs/1802.01666).

If you find this work useful in your research, please consider citing:

    @ARTICLE{elbanani_adviser_2018,
        author = "El Banani, Mohamed and Corso, Jason J.",
        title = "Adviser Networks: Learning What Question to Ask for Human-In-The-Loop Viewpoint Estimation",
        journal = ArXiv e-prints,
        year = 2018,
        month = Feb,
    }


If you have any questions, please email me at mbanani@umich.edu.


## To Do

- [ ] Import code from the viewpoint_estimation repository!
    - [x] Move Code!
    - [ ] Replicate results
- [ ] Run Adviser on the new attention-FT model
- [ ] Figure out a way of getting some proper priors on the data! Provide a more thorough analysis of the priors
- [ ] Compare against Bayesian, Information Theory, (and RL ?) approaches to this problem
- [ ] Look at the tasks described by the ICLR 2018 paper -- ["Ask The Right Question"](https://openreview.net/forum?id=S1CChZ-CZ)
- [ ] Move loggin inside of metrics

## Results

### Pascal3D - Vehicles with Keypoints

We fine-tuned both models on the Pascal 3D+ (Vehicles with Keypoints) dataset.
Since we suspect that the problem with the replication of the Click-Here CNN model
is in the attention section, we conducted an experiment where we only fine-tuned
those weights. As reported below, fine-tuning just the attention model achieves the best performance.

##### Accuracy
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN                | 89.26 | 74.36 | 81.93  | 81.85 |
| Render For CNN FT             | 93.55 | 83.98 | 87.30  | 88.28 |
| Render For CNN FT (reported)  | 90.6  | 82.4  | 84.1   | 85.7  |
| Click-Here CNN                | 86.91 | 83.25 | 73.83  | 81.33 |
| Click-Here CNN (reported)     | 96.8  | 90.2  | 85.2   | 90.7  |
| Click-Here CNN FT             | 92.97 | 89.84 | 81.25  | 88.02 |
| Click-Here CNN FT-Attention   | 94.48 | 90.77 | 84.91  | 90.05 |

##### Median Error
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN                | 5.16  | 8.53  | 13.46  | 9.05  |
| Render For CNN FT             | 3.04  | 5.83  | 11.95  | 6.94  |
| Render For CNN FT (reported)  | 2.93  | 5.63  | 11.7   | 6.74  |
| Click-Here CNN                | 4.01  | 8.18  | 19.71  | 10.63 |
| Click-Here CNN (reported)     | 2.63  | 4.98  | 11.4   | 6.35  |
| Click-Here CNN FT             | 2.93  | 5.14  | 13.42  | 7.16  |
| Click-Here CNN FT-Attention   | 2.88  | 5.24  | 12.10  | 6.74  |


### Pascal KP for Adviser  -- Baselines

#### Training Data with No Flip

##### Accuracy
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Worst                         | 99.63 | 99.26 | 95.59  | 98.16 |
| Mean                          | 100.  | 99.68 | 96.18  | 98.62 |
| Median                        | 100.  | 99.79 | 96.18  | 98.66 |
| Best                          | 100.  | 99.89 | 96.47  | 98.79 |


##### Median Error
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Worst                         | 0.    | 0.71  | 0.     | 0.24  |
| Mean                          | 0.    | 0.12  | 0.     | 0.04  |
| Median                        | 0.    | 0.    | 0.     | 0.0   |
| Best                          | 0.    | 0.    | 0.     | 0.0   |


#### Test Data with No Flip

#### Accuracy
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Worst                         | 87.19 | 81.4  | 75.81  | 81.47 |
| Mean                          | 89.68 | 84.73 | 78.17  | 84.19 |
| Median                        | 92.17 | 88.71 | 79.35  | 86.74 |
| Best                          | 95.37 | 92.37 | 87.91  | 91.88 |


#### Median Error
|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Worst                         |  3.7  | 7.1   | 13.98  | 8.26  |
| Mean                          |  3.41 | 6.65  | 13.21  | 7.76  |
| Median                        |  3.25 | 6.18  | 12.71  | 7.38  |
| Best                          |  2.92 | 5.45  | 11.69  | 6.69  |


## Acknowledgements

This work has been partially supported by DARPA W32P4Q-15-C-0070 (subcontract from SoarTech) and funds from the University of Michigan Mobility Transformation Center.
