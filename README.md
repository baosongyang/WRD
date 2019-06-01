# Word Reordering Detection
This word reordering detection task is based on the following paper:
* Assessing the Abilites of Self-Attention Networks to Learn Word Order. [Baosong Yang](https://baosongyang.site/), [Longyue Wang](http://www.longyuewang.com/), [Derek F. Wong](https://www.fst.um.edu.mo/en/staff/fstfw.html), Lidia S. Chao and [Zhaopeng Tu](http://zptu.net/). In ACL 2019.

## Introduction
The main purpose is to study how well the word order information learned by different neural networks. Specifically, we randomly move one word to another position, and examine whether a trained model can detect both the original and inserted positions. Our codes were built upon [THUMT-MT](https://github.com/THUNLP-MT/THUMT). We compare self-attention networks (SAN, [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)) with re-implemented RNN ([Chen et al., 2018](https://www.aclweb.org/anthology/P18-1008)), as well as directional SAN (DiSAN,[Shen et al., 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16126/16099)) that augments SAN with recurrence modeling.

## Citation
Please cite the following paper:
```
@inproceedings{yang2019assessing,
  author    = {Baosong Yang  and  Longyue Wang  and  Derek F. Wong  and Lidia S. Chao and Zhaopeng Tu},
  title     = {Assessing the Abilites of Self-Attention Networks to Learn Word Order.},
  booktitle = {ACL},
  year      = {2019}
}
```

## Usage

