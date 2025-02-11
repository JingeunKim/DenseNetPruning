# Pruning for Efficient DenseNet via Surrogate-Model-Assisted Genetic Algorithm considering Neural Architecture Search Proxies

This repository contains the code for the research paper titled "Pruning for Efficient DenseNet via Surrogate-Model-Assisted Genetic Algorithm considering Neural Architecture Search Proxies".

# Abstract
Recently, convolution neural networks have achieved remarkable progress in computer vision. These neural networks have a large number of parameters, which should be limited in resource-constrained environments. To address this problem, new pruning approaches have led to studies using neural architecture search (NAS) to determine optimal subnetworks. We propose a novel pruning framework using a surrogate model-assisted genetic algorithm considering NAS proxies (SMA-GA-NP). The DenseNet-BC (k=12) model was used as the baseline. For CIFAR-10 and CIFAR-100, with and without data augmentation, the error rate and number of parameters resulting from SMA-GA-NP were compared with those of the baseline model and other GA-based pruning methods. We achieved a highly competitive performance on CIFAR-10 compared with that obtained using other GA-based pruning methods and baselines. For CIFAR-10, compared with the baseline model, SMA-GA-NP achieved a notable 0.2%p to 0.5%p reduction in the error rate and reduced the number of parameters by 15\% to 16.25\%. We achieved a minimal performance degradation (<1%) with fewer parameters than the baseline model on CIFAR-100. For CIFAR-100, compared with the baseline model, SMA-GA-NP had a 0.39%p to 0.4%p increase in the error rate, while reducing the number of parameters by 17.5% to 18.75%. These findings highlight SMA-GA-NP's effectiveness in significantly reducing the number of parameters with a negligible impact on the model error rate. We also conducted an ablation study to explore the efficiency of the surrogate model and NAS proxies in SMA-GA-NP and identified the current limitations and future potential of SMA-GA-NP.


## Usage 
```
python main.py\
 --dataset <dataset>\
 --nClasses <number of classes>\
 --augmentation <augmentation>\
 --dropout <dropout rate>\
 --GA_epoch <epochs during GA>\
 --number_population <Popoulation size for GA>\
 --generations <Number of generations for GA>\
 --subset <ratio subset of training data>\
 --surrogate <the use of the surrogate model>
```



## Citation
If you find this useful, please cite the following paper:
```
TBA
```
