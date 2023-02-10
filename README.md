# HeteroRewire
The source code of Homophily-oriented Heterogeneous Graph Rewiring accepted by WWW 2023.

## Abstract

With the rapid development of the World Wide Web~(WWW), heterogeneous graphs~(HG) have explosive growth. Recently, heterogeneous graph neural network~(HGNN) has shown great potential in learning on HG. Current studies of HGNN mainly focus on some HGs with strong homophily properties~(nodes connected by meta-path tend to have the same labels), while few discussions are made in those that are less homophilous. Recently, there have been many works on homogeneous graphs with heterophily. However, due to heterogeneity, it is non-trivial to extend their approach to deal with HGs with heterophily. In this work, based on empirical observations, we propose a meta-path-induced metric to measure the homophily degree of a HG. We also find that current HGNNs may have degenerated performance when handling HGs with less homophilous properties. Thus it is essential to increase the generalization ability of HGNNs on non-homophilous HGs. To this end, we propose HDHGR, a homophily-oriented deep heterogeneous graph rewiring approach that modifies the HG structure to increase the performance of HGNN. We theoretically verify HDHGR. In addition, experiments on real-world HGs demonstrate the effectiveness of HDHGR.

## Citation

```
@inproceedings{Guo2023HeteroRewire,
    title={Homophily-oriented Heterogeneous Graph Rewiring},
    author={Jiayan Guo and Lun du and Wendong Bi and Qiang Fu and Xiaojun Ma and Xu Chen and Shi Han and Dongmei Zhang and Yan Zhang},
    booktitle={Proceedings of the ACM Web Conference {WWW}},
    year={2022}
}
```
