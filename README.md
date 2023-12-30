# ToThePoint

Pytorch implementation for ToThePoint results in the paper [ToThePoint: Efficient Contrastive Learning of 3D Point Clouds via Recycling](https://openaccess.thecvf.com//content/CVPR2023/html/Li_ToThePoint_Efficient_Contrastive_Learning_of_3D_Point_Clouds_via_Recycling_CVPR_2023_paper.html) by Xinglin Li, Jiajing Chen, Jinhui Ouyang, Hanhui Deng, Senem Velipasalar, Di Wu in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

<img src="./fig/framework.png#pic_center" width="900px" height="350px"/>


### Dependencies
python 3.8

Pytorch 1.7.1

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `***`
- `***`
- `***`


### Evaluation Results

 - 3D object classification
<p align="center">
<img src='./fig/classification1.png#pic_center' width="500px">
</p>

<p align="center">
<img src='./fig/classification2.png#pic_center' width="500px">
</p>

- Few-shot 3D object classification
<p align="center">
<img src='./fig/few_shot.png#pic_center' width="500px">
</p>

- 3D object part segmentation:

<p align="center">
<img src='./fig/part_seg.png#pic_center' width="500px">
</p>

- ablation study:

<p align="center">
<img src='./fig/ablation.png#pic_center' width="500px">
</p>



### Citing ToThePoint
If you find ToThePoint useful in your research, please consider citing:
BibTex:
```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Xinglin and Chen, Jiajing and Ouyang, Jinhui and Deng, Hanhui and Velipasalar, Senem and Wu, Di},
    title     = {ToThePoint: Efficient Contrastive Learning of 3D Point Clouds via Recycling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {21781-21790}
}
```
or
```
Xinglin Li, Jiajing Chen, Jinhui Ouyang, Hanhui Deng, Senem Velipasalar, Di Wu; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 21781-21790.
```

**Reference**

- [CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/papers/Afham_CrossPoint_Self-Supervised_Cross-Modal_Contrastive_Learning_for_3D_Point_Cloud_Understanding_CVPR_2022_paper.pdf) [[code]](https://github.com/MohamedAfham/CrossPoint)
