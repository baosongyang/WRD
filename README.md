# Word Reordering Detection
This word reordering detection task is based on the following paper:
* Assessing the Abilites of Self-Attention Networks to Learn Word Order. [Baosong Yang](https://baosongyang.site/), [Longyue Wang](http://www.longyuewang.com/), [Derek F. Wong](https://www.fst.um.edu.mo/en/staff/fstfw.html), Lidia S. Chao and [Zhaopeng Tu](http://zptu.net/). In ACL 2019.

## Introduction
The main purpose is to study how well the word order information learned by different neural networks. Specifically, we randomly move one word to another position, and examine whether a trained model can detect both the original and inserted positions. Our codes were built upon [THUMT-MT](https://github.com/THUNLP-MT/THUMT). We compare self-attention networks (SAN, [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)) with re-implemented RNN ([Chen et al., 2018](https://www.aclweb.org/anthology/P18-1008)), as well as directional SAN (DiSAN,[Shen et al., 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16126/16099)) that augments SAN with recurrence modeling.

## Citation
Please cite the following paper:
Baosong Yang, Longyue Wang, Derek F. Wong, Lidia S. Chao and Zhaopeng Tu. 2019. Assessing the Abilites of Self-Attention Networks to Learn Word Order. In ACL 2019.
