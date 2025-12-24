## Objective 
Evaluate the impact of data augmentation when training ResNet50 ([[1]]) on CIFAR-10.

## Data Augmentation Methods Used
- Color Jitter
- RandAugment ([[2]])
- Normalize (following ImageNet's mean and standard deviation)

## Training Details
- Optimizer: SGD + momentum + weight decay
- Criterion: CrossEntropyLoss
- Learning Rate Scheduler: Cosinne Annealing
- Number of epochs: 100

## Training Environment
CPU: AMD Ryzen 5 7600
GPU: AMD RX 5700XT
RAM: 16GB

## Final Accuracies
Testing: 86.45%
Training: 93.72%

## References
[[1]] He et al., *Deep Residual Learning for Image Recognition*, 2016
[[2]] Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*, 2020

[1]: https://arxiv.org/abs/1512.03385
[2]: https://arxiv.org/abs/1909.13719
