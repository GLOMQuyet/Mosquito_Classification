![logo](resources/logo.png)

# Author's thanks

This repository is inspired by https://github.com/Hawkeye-FineGrained/Hawkeye. Paper for this github https://arxiv.org/pdf/2310.09600.pdf. I am very grateful to authors Jiabei He, Yang Shen, Xiu-Shen We, Ye Wu for creating a common framework that can solve the Fine-Grained Image Recognition problem.

At the same time, I am also very grateful to authors Mehenika Akter, Mohammad Shahadat Hossain, Tawsin Uddin Ahmed, Karl Andersson of the paper ![Mosquito Classification using Convolutional
Neural Network with Data Augmentation](https://link.springer.com/chapter/10.1007/978-3-030-68154-8_74) provided me with the data to use for this project

# Hawkeye

Hawkeye is a unified deep learning based fine-grained image recognition toolbox built on PyTorch, which is designed for researchers and engineers. Currently, Hawkeye contains representative fine-grained recognition methods of different paradigms, including utilizing deep filters, leveraging attention mechanisms, performing high-order feature interactions, designing specific loss functions, recognizing with web data, as well as miscellaneous.

## Updates

**Nov 01, 2022:** Our Hawkeye is launched!

## Model Zoo

The following methods are placed in `model/methods` and the corresponding losses are placed in `model/loss`.

The table of experimental results for the following methods on `Mosquito` can be found in the `results.csv(Updating)` file.
Except for the asterisked methods, $300 \times 300$ input images were used.

- **Utilizing Deep Filters**
  - [S3N](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf)
  - [ProtoTree](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf)
  - [Interp-Parts](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Interpretable_and_Accurate_Fine-grained_Recognition_via_Region_Grouping_CVPR_2020_paper.pdf)
- **Leveraging Attention Mechanisms**
  - [MGE-CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.pdf)
  - [OSME+MAMC](https://arxiv.org/pdf/1806.05372v1)
  - [APCNN](https://arxiv.org/pdf/2002.03353.pdf)
- **Performing High-Order Feature Interactions**
  - [BCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf)
  - [CBCNN](https://arxiv.org/pdf/1511.06062)
  - [Fast MPN-COV](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf)
- **Designing Specific Loss Functions**
  - [API-Net](https://arxiv.org/pdf/2002.10191.pdf)
  - [Pairwise Confusion](https://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf)
  - [CIN](https://arxiv.org/pdf/2003.05235v1)
- **Recognition with Web Data**
  - [Peer-Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Webly_Supervised_Fine-Grained_Recognition_Benchmark_Datasets_and_an_Approach_ICCV_2021_paper.pdf)
- **Miscellaneous**
  - [NTS-Net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.pdf)
  - [CrossX](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf)
  - [DCL](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf)

## Get Started

We provide a brief tutorial for Hawkeye.

### Clone

```
git clone https://github.com/GLOMQuyet/Mosquito_Classification
cd Mosquito_Classification
```

### Make Environment
You can install Mosquito_Classification via pip, conda, or Docker(explanation below).
```
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```
OR
```
python -m venv .env
# Windows
.\.env\Scripts\activate
# Linux
source .env/bin/activate
```

### Requirements

```
pip install -r requirements.txt
```

####  Data Acquisition
As there is no standard dataset on mosquitoes available online, we had to construct the dataset from different online sources. We collected mosquito images
from websites like Pixabay , Getty Images , Shutterstock Images , iStock
etc. We collected approximately 40 images from Pixabay, 120 images from
Getty Images, 90 images from Shutterstock Images, 60 images from iStock and
the rest from other sources. We had a total of 442 images; 188 of aedes species,
130 of anopheles species and 124 of culex species. Image displays some sample
images of our dataset. We took help from some sources like  to label the data
correctly.

![sample](resources/sample.png)


#### Characteristics of Aedes, Anopheles and Culex
There are some properties associated with the mosquitoes by which one can
recognize the mosquitoes and differentiate between them. Image shows example
images of the three mosquito species: aedes, anopheles and culex. The characteristics of aedes, anopheles and culex are given below

Aedes. Aedes mosquitoes can be identified differently as they possess black and
white markings all over the body. These mosquitoes stay awake in the daytime in
dark corners. Primarily female aedes bites humans and sucks blood so that they
can lay eggs. This mosquito is the carrier for infectious diseases like dengue,
chikungunya etc. These diseases are mediated to humans by the bites of an
affected female aedes

Anopheles. The body color of an anopheles mosquito’s body is brown or black.
It consists of 3 body parts: the head, abdomen and thorax. The lower body of
the vector points to the top while they are resting. It lays eggs after sucking
blood. Even though it can live some weeks to a month, it is able to produce
eggs in that time span. The anopheles mosquito is considered throughout the
world for bearing one of the most infectious diseases called malaria. It is also
responsible for heartworm

Culex. Culex appears to be a black mosquito with some white stripes on some
body parts. Male and female culex, both of them, live on honey and herb liquids.
When a female culex is willing to produce eggs, it feeds on the blood of humans,
other beasts and also birds. It is compulsory for the female culex to have blood
so that they can reproduce. Though the female mosquitoes bite only the birds
at some point, they also attack the mammals sometimes. This culex mosquito is
responsible for spreading the Zika virus. It is also found in charge of spreading
west nile virus, encephalitis and filariasis

![mosquito](resources/mosquito.png)


#### Data processing and data augmentation

> First, decompress the data. If you use Windows, just go to the `data/dataset.zip` file to decompress them. From there we will have the `data/dataset` directory.
If you use linux, you can use the following command:
```
!unzip data/dataset.zip
```
> Second (optional) for data augmentation case command:
```
python dataset/data_augmentation.py
```
They will create a folder named `data/dataset_mosquito`

We provide the meta-data file of the datasets in `metadata/`, and the train list and the val list are also provided according to the  official splittings of the dataset. There is no need to modify the decompressed directory of the dataset. The following is an example of the directory structure of two datasets.

> Finally the metadata file of the datasets is in the form `metadata/`, and the training list and val list are also provided according to the formal splits of the dataset. There is no need to modify the extracted folder of the dataset. The following is the code that creates the metadata file and an example of the directory structure of the two data sets.
```
python dataset\metada.py
```
```
data
├── dataset
│   ├── Aedes
│   │   
│   │        
│   │── Anopheles
│   │   
│   │   
│   └── Culex
├── dataset_mosquito
│   ├── Aedes
│   │   
│   │     
│   │      
│   ├── Anopheles
│   │   
│   │      
│   │     
│   │   
│   └── Culex
metadata
├── train.txt
│
└── val.txt
```

#### Configuring Datasets

When using different datasets, you need to modify the dataset path in the corresponding config file. `meta_dir` is the path to the meta-data file which contains train list and val list. `root_dir` is the path to the image folder in `data/`. Here are two examples.

> Note that the relative path in the meta-data list should match the path of `root_dir`. 

```
dataset:
  name: Mosquito
  root_dir: data/dataset_mosquito
  meta_dir: metadata/dataset_mosquito
```



> Note that, for [ProtoTree](https://github.com/M-Nauta/ProtoTree), it was trained on an offline augment dataset, refer to the [link](https://github.com/M-Nauta/ProtoTree#data) if needed. We just provide meta-data for the offline augmented mosquito in `metadata/dataset_mosquito`.

### Training

For each method in the repo, we provide separate training example files in the `Examples/` directory.

- For example, the command to train an APINet:

  ```bash
  python Examples/APINet.py --config configs/APINet.yaml
  ```

  The default parameters of the experiment are shown in `configs/APINet.yaml`.

Some methods require multi-stage training. 

- For example, when training BCNN, two stages of training are required, cf. its two config files.

  First, the first stage of model training is performed by:

  ```bash
  python Examples/BCNN.py --config configs/BCNN_S1.yaml
  ```

  Then, the second stage of training is performed later. You need to modify the weight path of the model (`load` in `BCNN_S2.yaml`) to load the model parameters obtained from the first stage of training, such as `results/bcnn/bcnn_cub s1/best_model.pth`.

  ```bash
  python Examples/BCNN.py --config configs/BCNN_S2.yaml
  ```

In addition, specific parameters of each method are also commented in their configs.

### Testing

We provide sample codes to test a model, you can run the command to test BCNN:

```bash
python test.py --config configs/test.yaml
```

You can modify `test.py` and `test.yaml` to test other models.

## License

This project is released under the [MIT license](./LICENSE).

## Contacts

If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: [weixs.gm@gmail.com](mailto:weixs.gm@gmail.com)

Jiabei He: [hejb@njust.edu.cn](mailto:hejb@njust.edu.cn)

Yang Shen: [shenyang_98@njust.edu.cn](mailto:shenyang_98@njust.edu.cn)

## Acknowledgements

This project is supported by National Key R&D Program of China (2021YFA1001100), National Natural Science Foundation of China under Grant (62272231), Natural Science Foundation of Jiangsu Province of China under Grant (BK20210340), and the Fundamental Research Funds for the Central Universities (No. 30920041111, No. NJ2022028).
