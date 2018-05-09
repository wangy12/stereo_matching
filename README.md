# Reimplementing "Efficient Deep Learning for Stereo Matching" in TensorFlow and Keras

---


[//]: # (Image References)

[image1]: ./disp_images90/disp_map_000118_10.png "Undistorted chessboard"
[image2]: ./disp_images225Keras/disp_map_000118_10.png "Undistorted road"
[image3]: ./disp_images2600/disp_map_000118_10.png "straight1"
[image4]: ./disp_images4820/disp_map_000118_10.png "curve3"
[image5]: ./originalImg/000118_10.png
[image6]: ./originalImg/Plot1.png

A CNN-based siamese networks for stereo matching is proposed in the paper [Efficient Deep Learning for Stereo Matching](https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf) .

The authors of the paper provided the [code](http://www.cs.toronto.edu/deepLowLevelVision/) in lua. A tensorflow reimplementation by Algolux engineers/researchers can be found [here](https://github.com/fjulca-aguilar/dl_stereo_matching).

My work is based on the above tensorflow pipeline. As an assignment, I contributed to the following two parts:

1. Reimplemented functions 'win9_dep4.py' and 'win29_dep9.py', which correspond to the deep networks 'Ours(9)' and 'Ours(29)' in the paper,  in tensorflow.

2. Reimplemented the CNN networks for architectures $n=9,19,29,37$ in Keras and made Keras as part of the tensorflow workflow. Note that I used Keras to build models, and wrote optimizing codes and all the others in tensorflow following [Algolux's work](https://github.com/fjulca-aguilar/dl_stereo_matching).

I ran and tested both reimplementations, however, I could not run as many iterations (e.g., 40000 iterations) as possible because my laptop is not very powerful and is always occupied by my PhD research.
Finally, 4800 steps are processed for the tensorflow pipeline and 200 steps for the Keras pipeline.

Since the training iteration number is so small (4800 vs 40000), the stereo matching performance is not very satisfying. Testing images and qualitative estimates are far from the results in the paper.

In the following, the loss curve for 4800 iterations and the testing images after *iter = 90,2600,4800* are shown. Since I stoped and restored the program several times, these test images were kept. 

The loss curve shows the deep learning framework actually converges. If more powerful GPUs are available, we can train the deep neraul network for $40000$ more iterations. In that case, the result will be similar to those in the paper.

It can be seen from these images that with the increase of the number of iteration, the stereo matching effect by deep learning becomes obvious. From the images of 2600 and 4820 training iterations, we can see the road and cars. The closer objects have a lighter color than the objects far away. We know that after 40000 or even more training steps, there will be less noise and the stereo estimation will be more accurate (the loss will be much smaller.)


---

![alt text][image5]

---

**Iter=90** (kitti2012 + 'win19_dep9/Ours(19)' in TensorFlow)
![alt text][image1]

---

**Iter=200** (kitti2012 + 'win19_dep9/Ours(19)' in Keras)
![alt text][image2]

---

**Iter=2600** (kitti2012 + 'win19_dep9/Ours(19)' in TensorFlow)
![alt text][image3]

---

**Iter=4800** (kitti2012 + 'win19_dep9/Ours(19)' in TensorFlow)
![alt text][image4]

---

**Loss vs Iteration** (kitti2012 + 'win19_dep9/Ours(19)' in TensorFlow)
![alt text][image6]



References:

http://www.cs.toronto.edu/deepLowLevelVision/papers/cvpr16/top.pdf

https://bitbucket.org/saakuraa/cvpr16_stereo_public/src

https://github.com/fjulca-aguilar/dl_stereo_matching

