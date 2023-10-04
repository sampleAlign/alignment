# A general image misalignment correction method for tomography experiments
**Zhen Zhang, Zheng Dong, Hanfei Yan, Ajith Pattammattel, Xiaoxue Bi, Yuhui Dong, Gongfa Liu, Xiaokang Sun & Yi Zhang**

### Summary
We introduce a novel alignment method (OCMC) for tomography datasets that utilizes the sample's outer contour structure to correct joint offsets between projection images. This method has demonstrated high effectiveness in correcting misalignment in both simulated and experimental datasets obtained from various imaging techniques. Furthermore, the proposed method can be integrated into the data acquisition pipeline, enabling real-time data processing.

![Fig2](https://github.com/sampleAlign/alignment/assets/102127175/b9f5cdd4-00ff-448d-a768-3a0f52c92885)

### The repository includes
- OCMC code
- OCS extractor(default) code
- Data for test
- Jupyter notebooks to visualize the alignment pipeline at every step

*New algorithm that take into account properties out the field of view is coming soon, stay tuned.*

# Installation
1. Clone this repository via git clone https://github.com/sampleAlign/alignment.git
2. Install dependencies and current repo `pip install -r requirements.txt`

# Run Jupyter notebooks

### Quick demo
Open demo.ipynb. You can use the example data to view the OCMC alignment steps and results.
![Fig4v2](https://github.com/sampleAlign/alignment/assets/102127175/5de208f9-4dc2-48fc-90e2-dbfe70279032)

# Citation
Use this bibtex to cite this repository:
```
@article{Zhang2023,
  title={A general image misalignment correction method for tomography experiments},
  author={Zhen Zhang, Zheng Dong, Hanfei Yan, Ajith Pattammattel, Xiaoxue Bi, Yuhui Dong, Gongfa Liu, Xiaokang Sun and Yi Zhang},
  journal={iScience},
  year={2023},
  volume={26},
  issue={10},
  doi={10.1016/j.isci.2023.107932},
  publisher={Cell Press}
}
```
# Contributing
Contributions to this repository are always welcome.

# Requirements
Python 3.9, numpy 1.22, and other common packages listed in `requirements.txt`

# Other
Zhen zhang is looking for a **postdoctoral** position with a focus on computed tomography methodology (algorithms, deep learning, etc.), please contact `zznsrl@mail.ustc.edu.cn` if you are interested.




