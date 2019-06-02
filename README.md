# Word Reordering Detection
This word reordering detection task (**WRD**) is based on the following paper:
* Assessing the Ability of Self-Attention Networks to Learn Word Order. [Baosong Yang](https://baosongyang.site/), [Longyue Wang](http://www.longyuewang.com/), [Derek F. Wong](https://www.fst.um.edu.mo/en/staff/fstfw.html), Lidia S. Chao and [Zhaopeng Tu](http://zptu.net/). In ACL 2019.

## Introduction
The main purpose is to study how well the word order information learned by different neural networks. Specifically, we randomly move one word to another position, and examine whether a trained model can detect both the original and inserted positions. Our codes were built upon [THUMT-MT](https://github.com/THUNLP-MT/THUMT). We compare self-attention networks (SAN, [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)) with re-implemented RNN ([Chen et al., 2018](https://www.aclweb.org/anthology/P18-1008)), as well as directional SAN (DiSAN,[Shen et al., 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16126/16099)) that augments SAN with recurrence modeling.

## Citation
Please cite the following paper:
```
@inproceedings{yang2019assessing,
  author    = {Baosong Yang  and  Longyue Wang  and  Derek F. Wong  and Lidia S. Chao and Zhaopeng Tu},
  title     = {Assessing the Ability of Self-Attention Networks to Learn Word Order},
  booktitle = {ACL},
  year      = {2019}
}
```
## Data
We conduct this task on the English sentences, which are extracted from the WMT14 En⇒De data with maximum length to 80. For each sentence in different sets (i.e. training,  validation,  and test sets), we construct an instance by randomly moving a word to another position. Finally we construct 7M, 10Kand 10K samples for training, validating and testing, respectively. Note that a sentence can be sampled multiple times, thus each dataset in the WRD data contains more instances than in the machine translation data.

For other languages, we provide scripts for generating such kind of corpus "./script/reorder_word.py" and recover it to the original format "./script/recover_order.py".
## Usage
* This program is based on [THUMT-MT](https://github.com/THUNLP-MT/THUMT). We add options for running RNN- and DiSAN-based models which are named "**rnnp**" and "**transformer_di**", respectively. To run machine translation models, you may read the documentation of the original implementation.  
* **To examine pre-trained MT encoders** on WRD task: 1. put your model checkpoint files under the "eval" folder; 2. we provide an example script "word_order_MT.sh" to assess the ability of SAN to learn word order, you can evaluate other models by modifying the example script.
* **To examine randomly initialized encoders** on WRD task: 1. put your well-trained MT models under the "eval" folder (merely use word embeddings, you can also choose other well-trained word embeddings); 2. we provide an example script "word_order_MT.sh" to assess the ability of SAN to learn word order, you can evaluate other models by modifying the example script. **Note that**, if you use word embeddings in pre-trained MT models, please remember to rename the scope name in the model file, making the WRD model fail to load existing parameters and re-initialize new parameters, for example: modify: ./thumt/models/transformer.py:
```
Line 48: "encoder" => "encoder2"
```
* **To assess the accuracy of models**: you can use our scripts released in ./scripts/
* **Effect of wrong word order noises**:  we make erroneous word order noises on WMT14 En-De development set by moving one word to another position, and evaluate the drop of the translation quality of each model. The data and script can be found in "./robustness"
