## Predicting Alzheimer's disease progression trajectory and clinical subtypes using machine learning
### Authors

Vipul K. Satone, Rachneet Kaur, Hampton Leonard, Hirotaka Iwaki, Lana Sargent, for the Alzheimer’s Disease Neuroimaging Initiative, Sonja W. Scholz, Mike A. Nalls, Andrew B. Singleton, Faraz Faghri, Roy H. Campbell  

#### [Interactive website](https://share.streamlit.io/kaurrachneet6/mlforadni/main) for clinical researchers to predict the clinical subtype of an Alzheimer's disease patient based on clinical parameters

## Abstract
### Background
Alzheimer’s disease (AD) is a common, age-related, neurodegenerative disease that impairs a person's ability to perform day-to-day activities. Diagnosing AD is challenging, especially in the early stages. Many patients remain undiagnosed, partly due to the complex heterogeneity in disease progression. This diagnostic challenge highlights a need for early prediction of the disease course to assist its treatment and tailor management to the disease progression rate. Recent developments in machine learning techniques provide the potential to predict disease progression and trajectory of AD and classify the disease into different etiological subtypes. 

### Methods and findings
The suggested work clusters participants in distinct and multifaceted progression subgroups of AD and discusses an approach to predict the progression stage from baseline diagnosis. We observe that the myriad of clinically reported symptoms summarized in the proposed AD progression space corresponds directly to memory and cognitive measures, classically been used to monitor disease onset and progression. The proposed work concludes notably accurate prediction of disease progression after four years from the first 12 months of post-diagnosis clinical data (Area Under the Curve of 0.92 (95% confidence interval (CI),  0.90-0.94), 0.96 (95% CI, 0.92-1.0), 0.90 (95% CI, 0.86-0.94) and 0.83 (95% CI, 0.77-0.89)  for controls, high, moderate and low progression rate patients respectively). Further, we explore the long short-term memory (LSTM) neural networks to predict the trajectory of a patient’s progression. 

### Conclusion
The machine learning techniques presented in this study may assist providers with identifying different progression rates and trajectories in the early stages of disease progression, hence allowing for more efficient and unique care deliveries. With additional information about the progression rate of AD at hand, providers may further individualize the treatment plans. The predictive tests discussed in this study not only allow for early AD diagnosis but also facilitate the characterization of distinct AD subtypes relating to trajectories of disease progression. These findings are a crucial step forward to early disease detection. Additionally, models can be used to design improved clinical trials for AD research.

* Note: The versions of python the packages used in the analysis can be found in the file python_package_versions.txt.
* Code for the [interactive webpage](https://share.streamlit.io/kaurrachneet6/mlforadni/main) is available on https://github.com/kaurrachneet6/MLforADNI.git

## Code description:
Folder **Version 1 ML4H@NeurIPS2018** contains codes for results presented in 
```
@article{satone2018learning,
  title={Learning the progression and clinical subtypes of Alzheimer's disease from longitudinal clinical data},
  author={Satone, Vipul and Kaur, Rachneet and Faghri, Faraz and Nalls, Mike A and Singleton, Andrew B and Campbell, Roy H},
  journal={arXiv preprint arXiv:1812.00546},
  year={2018}
}
```

Main folder contains codes for results presented in 
```
@article{satone2019predicting,
  title={Predicting Alzheimer’s disease progression trajectory and clinical subtypes using machine learning},
  author={Satone, Vipul K and Kaur, Rachneet and Leonard, Hampton and Iwaki, Hirotaka and Sargent, Lana and Scholz, Sonja W and Nalls, Mike A and Singleton, Andrew B and Faghri, Faraz and Campbell, Roy H and others},
  journal={bioRxiv},
  pages={792432},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```
