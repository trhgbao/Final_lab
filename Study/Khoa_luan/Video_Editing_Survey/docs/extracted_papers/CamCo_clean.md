The rapid evolution of diffusion-based generative models [38, 39, 37, 36, 3] has empowered general users to generate video content from textual or visual inputs easily.
Recent advancements in video generative models [23, 4, 1, 14] have underscored the importance of user control [52, 14, 13, 32] over the generated content. Rather than relying on the cumbersome and error-prone process of prompt engineering, controllable generation techniques ensure that the output adheres to more specified and fine-grained control signals, thereby enhancing user satisfaction by producing more desirable outcomes.
Recent studies [59, 24, 29] have incorporated additional training layers to integrate these control signals, further refining content generation capabilities.

Despite the diverse control signals available (e.g., depth, edge map, human pose) [52, 14, 13, 32, 59, 24, 29], controlling camera viewpoints in generated content has received little attention.
Camera motion, a crucial filmmaking technique [30], enables content creators to shift the audience’s perspective without cutting the scene [44], thereby conveying emotional states effectively [56]. This technique is vital for ensuring that videos are practically usable in downstream applications such as augmented reality, filmmaking, and game development [16]. It allows creators to communicate more dynamically with their audience and adhere to a pre-designed script or storyboard.

To enable camera control in content generation, early methods [14, 13] utilize a fixed number of categories to represent camera trajectories, providing a coarse control over the camera motion. They use LoRA [19] to fuse the camera information and conditionally generate videos that belong to certain categories of camera motion. To improve the camera motion granularity, [51] has proposed to use adapter layers to accept normalized camera extrinsic matrices.
However, this approach uses one-dimensional numeric values to represent the camera pose, which struggles to precisely control the video content when provided with complex camera motion.

To overcome the above issues, we introduce CamCo, a camera-controllable image-to-video generation framework that produces 3D-consistent videos.
Our framework is built upon a pre-trained image-to-video diffusion model, preserving the majority of the original model parameters to maintain its generative capabilities.
To enhance the accuracy of camera motion control, we represent camera pose with Plücker coordinates that encode both camera intrinsics and extrinsics into a pixel-wise embedding, which is a dense conditioning signal for video generation.
Moreover, to improve the geometric consistency of synthesized videos, we introduce a new epipolar constraint attention module in each attention block to enforce epipolar constraints across frames.
Finally, we implement a data curation pipeline that annotates in-the-wild video frames with estimated camera poses, enhancing our capability of generating videos with large object motions.
Our experiments on various domains demonstrate that CamCo outperforms previous state-of-the-art methods in terms of visual quality, camera controllability, and geometry consistency.

Our contributions can be summarized as follows,

•

We propose CamCo, a novel camera-controllable image-to-video generation framework that can generate high-quality, 3D-consistent videos.

•

We build the first 3D-consistent video diffusion model by adapting a pre-trained image-to-video diffusion model. We parameterize camera information via Plücker coordinates and incorporate epipolar constraints via new epipolar constraint attention modules.

•

We introduce a data curation pipeline to handle in-the-wild videos with dynamic subjects and fine-tune CamCo on the curated dataset to enhance its ability to generate videos with both camera ego-motion and dynamic subjects.

•

Our method exhibits superior 3D consistency, visual quality, and camera controllability when compared with previous works.

We propose CamCo, a novel camera-controllable image-to-video generation framework that can generate high-quality, 3D-consistent videos.

We build the first 3D-consistent video diffusion model by adapting a pre-trained image-to-video diffusion model. We parameterize camera information via Plücker coordinates and incorporate epipolar constraints via new epipolar constraint attention modules.

We introduce a data curation pipeline to handle in-the-wild videos with dynamic subjects and fine-tune CamCo on the curated dataset to enhance its ability to generate videos with both camera ego-motion and dynamic subjects.

Our method exhibits superior 3D consistency, visual quality, and camera controllability when compared with previous works.

2.1 Diffusion-based Video Generation

The recent advances in diffusion models [38, 39, 37, 36, 3] have provided users with enhanced flexibility in visualizing their imaginations. Leveraging large-scale image-text paired datasets [42, 6], diffusion models have become the state-of-the-art generators for text-to-image (T2I) synthesis. However, due to the lack of high-quality video-text datasets [1, 4], researchers have extensively explored adapting text-to-image generators into text-to-video (T2V) generators.
Text2Video-Zero [23] builds the first training-free video generation pipeline. AnimateDiff [14] constructs a motion module applicable to various base T2I models. VideoLDM [4] and SVD [1] inflate the T2I models with additional temporal layers and train on curated datasets to improve the visual quality.
The recently released Sora model [5] has demonstrated impressive video generation results, drawing attention to transformer-based diffusion backbones [31, 27, 57, 58].
Our work builds upon the open-source SVD [1] model. We introduce novel camera conditioning and geometry-aware blocks to achieve camera controllability and geometry consistency for the challenging image-to-video generation task.

2.2 Controllable Content Creation

Along with the rapid development of generative models that produce content from various input modalities, improving user control over generation has also attracted significant attention.
ControlNet [59], T2I-adapter [29], and GLIGEN [24] pioneered the introduction of control signals, including depth, edge maps, semantic maps, and object bounding boxes, to text-to-image generation. These approaches ensure the synthesis backbone [38] remains intact while adding trainable modules to maintain both control and generative capabilities.
Similar approaches have been applied to video generation, supporting controls like depth, bounding boxes, and semantic maps [14, 13, 52, 32]. However, controlling camera motion has received limited attention.
AnimateDiff [14] and SVD [1] explore class-conditioned video generation, clustering camera movements, and using LoRA [19] modules to generate specific camera motions. MotionCtrl [51] enhances control by employing camera extrinsic matrices as conditioning signals. While effective for simple trajectories, their reliance on 1D numeric values results in imprecise control in complex real-world scenarios.
Our work significantly improves granularity and accuracy in camera control by introducing Plücker coordinates and geometry-aware layers to image-to-video generation. Concurrently, CameraCtrl [15] also employs Plücker coordinates but focuses on the text-to-video generation while paying less attention to 3D consistency.

2.3 3D Scene Synthesis

Due to the lack of high-quality 3D scene datasets [10, 9], it is challenging to adapt the success of image and video generation [3, 37, 38, 39, 57, 58] into 3D generation tasks [26, 43]. DreamFusion [35] addresses this issue by introducing score distillation sampling, which distills knowledge from 2D diffusion models.
This technique has been successfully applied to 3D object synthesis from text [25, 46, 50] and image [53, 26, 43] input.
However, due to the limited ability of 2D diffusion models to generate complex scenes, score distillation is not directly applicable to scene generation. Consequently, compositional generation [33, 60, 54, 12] has emerged as a popular approach to tackle these challenges.
Video generators [49, 28, 63]
have also been adapted to generate simple objects by conditioning the three degrees of freedom (DoF) camera information on the time embeddings of diffusion models.
However, these approaches are not directly applicable to real-world scenarios because their simplifications of camera information do not suffice for real videos that contain diverse extrinsic and intrinsic parameters.
Our work focuses on generating 3D-consistent videos from in-the-wild images, providing insights into the challenging direction of 3D scene synthesis.

In this section, we present our novel method for generating camera-controllable, geometry-consistent videos, as shown in Fig. 2. First, we outline the preliminaries of the pre-trained image-to-video diffusion model. Then, we introduce our camera parameterization and control modules, which integrate camera control into the pre-trained video diffusion model. Further, we describe our epipolar constraint attention mechanism, ensuring the geometric consistency of generated videos. Finally, we discuss how to curate data annotations to generate object motion effectively.

3.1 Image-to-video Generation

The task of image-to-video generation involves using a single image I0I_{0} as input of a video generator to produce a sequence of output frames O1,⋯,OnO_{1},\cdots,O_{n} with temporal consistency. In our experiments, nn is typically set to 14. To add camera control, we retain the original input and output format while additionally introducing a set of camera information C1,⋯,CnC_{1},\cdots,C_{n}. These camera parameters Ci{C_{i}} are used as fine-grained conditioning of the video generator, ensuring that the generated frames follow the viewpoint changes specified by the camera sequence.

Base Model Architecture

We train our CamCo based on a publicly available pre-trained image-to-video diffusion model, Stable Video Diffusion (SVD) [1]. SVD is built upon Stable Diffusion 2.1 model [38] that is originally trained through an EDM [22] framework.
SVD inserts temporal convolution and attention layers after spatial layers following VideoLDM [4].
Unlike previous works that only train the temporal layers [4, 14] or are training-free [23], SVD fine-tunes all parameters.

SVD incorporates a continuous-time noise scheduler [22], which supports a continuous range of noise levels. The diffusion model learns to gradually denoise a high-variance Gaussian noise towards the (video) data distribution 𝐱0∼p0\mathbf{x}_{0}\sim p_{0}. Let p⁡(𝐱,σ⁡(t))p(\mathbf{x};\sigma(t)) denote the marginal probability of noisy data 𝐱t=𝐱0+𝐧⁡(t)\mathbf{x}_{t}=\mathbf{x}_{0}+\mathbf{n}(t) where the added Gaussian noise 𝐧⁡(t)∼𝒩⁡(0,σ2​(t)​𝐈)\mathbf{n}(t)\sim\mathcal{N}(0,\sigma^{2}(t)\mathbf{I}), the iterative denoising process corresponds to the probability flow ordinary differential equation (ODE):



d​𝐱=−σ˙​(t)​σ​(t)​∇𝐱​log⁡p⁡(𝐱,σ⁡(t))​d​t,d\mathbf{x}=-\dot{\sigma}(t)\sigma(t)\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma(t))dt,

(1)

where ∇𝐱​log​p​(𝐱,σ⁡(t))\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma(t)) is the score function parameterized by a neural network D𝜽D_{\boldsymbol{\theta}} through
∇𝐱​log​p​(𝐱,σ)≈(D𝜽​(𝐱,σ)−𝐱)/σ2\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma)\approx\left(D_{\boldsymbol{\theta}}(\mathbf{x};\sigma)-\mathbf{x}\right)/\sigma^{2}. The training objective is denoising score matching:



𝔼⁡[‖D𝜽​(𝐱0+𝐧,σ,𝐜)−𝐱0‖22].\mathbb{E}\left[\left\|D_{\boldsymbol{\theta}}\left(\mathbf{x}_{0}+\mathbf{n};\sigma,\mathbf{c}\right)-\mathbf{x}_{0}\right\|_{2}^{2}\right].

(2)

where 𝐜\mathbf{c} denotes the conditioning information (e.g., the input image I0I_{0}).

Classifier-free guidance [18] (CFG) is also adopted for better conditioning following.
During training, the condition signal 𝐜\mathbf{c} is randomly set to a zero tensor ∅\emptyset to properly learn an unconditional model D⁡(𝐱,∅)D(\mathbf{x};\emptyset) with a 10% probability.
At inference time, CFG is formulated as follows,



Dω​(𝐱,𝐜)=ω⁡(D⁡(𝐱,𝐜)−D⁡(𝐱,∅))+D⁡(𝐱,∅),D_{\omega}(\mathbf{x};\mathbf{c})=\omega(D(\mathbf{x};\mathbf{c})-D(\mathbf{x};\emptyset))+D(\mathbf{x};\emptyset),

(3)

where ω\omega is the weighting coefficient. Following SVD [1], ω\omega is empirically set to linearly increase from 1 for the first frame and 3 for the last frame.

3.2 Injecting Camera Control to Video Diffusion Model

Camera Parameterization

Unlike generating canonicalized 3D objects that operate in a 3-DoF camera space, video diffusion models dealing with real-world dynamic scenes must manage 6-DoF cameras with diverse intrinsic parameters. Consequently, representing camera information using elevation [49] or camera extrinsics [51] is suboptimal. Moreover, using one-dimensional numeric values for camera extrinsic as conditioning leads to imprecise camera control [51] in challenging camera trajectories.
To comprehensively represent both camera extrinsic and intrinsic information, we draw inspiration from light field networks [45] and adopt Plücker coordinate  [20]. Specifically, let oo be the ray origin and dd be the ray direction of a pixel. The Plücker coordinate is formulated as P=(o×d′,d′)P=(o\times d^{\prime},d^{\prime}), where ×\times represents the cross product and d′d^{\prime} is the normalized ray direction d′=d‖d‖d^{\prime}=\frac{d}{||d||}. Given camera extrinsics E=[𝐑|𝐓]E=[\bf{R}|\bf{T}] and intrinsics 𝐊\bf{K},
the ray direction du,vd_{u,v} for 2D pixel located at (u,v)(u,v) is defined as d=𝐑𝐊−𝟏​(𝐮𝐯𝟏)+𝐓d=\bf{R}\bf{K}^{-1}(\begin{smallmatrix}u\\
v\\
1\end{smallmatrix})+\bf{T}.
All camera poses are defined relatively to the first frame.
Plücker coordinates uniformly represent rays, making them a suitable positional embedding for the network, and they have proven successful in parameterizing 360-degree unbounded light fields [45].

Camera Control Module

To incorporate the camera embeddings, we add a simple adapter layer to each temporal attention block in the pre-trained video generator.
This design best preserves the generation abilities the base model acquired during pre-training [59, 29, 24].
Specifically, in each temporal attention block, we first
concatenate the Plücker coordinates with the network features along the channel dimension, and then pass them into the 1×11\times 1 convolution layer to project back to the original feature space. The projected features are passed to the remaining temporal attention layers.
Plücker embeddings are downsampled via interpolation if their spatial resolution mismatches with the network features at bottleneck layers.
Inspired by ControlNet [59], we zero-initialize partial weights of the convolution layer so that at the start of training, the whole network remains unaffected by these additional layers, and we gradually add the camera control through the gradient descent.

3.3 Ensuring 3D-Consistent Generation

Drawback of vanilla architecture

While using the Plücker coordinates results in more fine-grained camera control, it does not directly ensure the geometric consistency of the generated videos.
This inconsistency issue is primarily rooted in the inefficiency of the vanilla architecture of video diffusion models (e.g., SVD) in modeling geometric relationships across frames.
Recall that the vanilla architecture introduces dense self-attention in spatial and temporal dimensions, where any pixel at any frame is free to attend to any other pixels, potentially leading to pixel-copying behaviors [21] that appear correct but do not adhere to geometric constraints.
To address this issue, we need to design an attention masking mechanism to ensure each pixel attends to features that are geometrically correlated.

Epipolar Constraint Attention (ECA)

Due to the missing depth information from the in-the-wild input images, we do not have the exact pointwise correspondences across frames. Nevertheless, for each pixel in the target view, we can derive an epipolar line in the source view. Epipolar constraint describes that the corresponding point of a feature in one image must lie on the epipolar line associated with that feature’s projection in another image.
We then propose epipolar constraint attention (ECA), which conducts cross-attention between the features on the epipolar line and features at the target location, ensuring that the current frames adhere to projective geometry. We visualize our ECA block in Fig. 2(b) and provide more details in Sec. A.

Our epipolar constraint attention associates the source view O1O_{1} with the target views Oi,i∈[2,n]O_{i},i\in[2,n] through the epipolar lines.
Due to the diverse camera pose sequences, the epipolar lines can vary in direction and length, posing a challenge for constructing epipolar lines that efficiently support batch attention calculation. To ensure efficiency, we propose constructing the epipolar lines by resampling the visible pixel locations, thereby avoiding variable-length epipolar line features.

At timestep tt, we represent zti∈Rh​w×dz_{t}^{i}\in\mathrm{R}^{hw\times d} as the latent feature of ii-th frame and Eti∈Rh​w×l×dE_{t}^{i}\in\mathrm{R}^{hw\times l\times d} as the point features along the corresponding epipolar lines, sampled from the feature map of the first frame, where ll is the number of points on the epipolar line.
Denote the query, key, and value by q:=zt1​Wq∈ℝh​w×1×dq:=z_{t}^{1}W_{q}\in\mathbb{R}^{hw\times 1\times d}, k:=Eti​Wk∈ℝh​w×l×dk:=E_{t}^{i}W_{k}\in\mathbb{R}^{hw\times l\times d}, and v:=Eti​Wv∈ℝh​w×l×dv:=E_{t}^{i}W_{v}\in\mathbb{R}^{hw\times l\times d}, respectively, where WqW_{q}, WkW_{k} and WvW_{v} are the projection matrices. Then, ECA between frame ii and the first frame is given by
ECA​(zti,Eti)=σ⁡(q​kTd)​v∈ℝh​w×d\text{ECA}(z_{t}^{i},E_{t}^{i})=\sigma(\frac{qk^{T}}{\sqrt{d}})v\in\mathbb{R}^{hw\times d}.
Unlike previous works that apply spatial binary masks [21] or weight maps [47] to the original cross-attention in O⁡((h​w)2)O((hw)^{2}), our attention mechanism implements the cross-attention matrix of size O⁡(h​w​l)O(hwl).
This efficient design avoids unnecessary computation and improves the efficiency of imposing the epipolar constraint by one order of magnitude (assuming l≈O⁡(h)≈O⁡(w)l\approx O(h)\approx O(w)), which is crucial for video generation where multiple frames are generated simultaneously.
For training, we only update the weights in the ECA layers while freezing the base video model. This preserves the base model’s generation and generalization abilities.

3.4 Improving Object Motion

While the above design enforces good camera control and geometric consistency, the model can easily overfit to generate only static scenes with little object motion because it lacks the training data with dynamic scenes.

Augmented Training Dataset with better Motion

To overcome this issue, we need to augment our training dataset with dynamic videos that contain rich object motion. However, the number of available dynamic videos with camera poses is greatly limited due to the difficulties in collecting annotations. Consequently, we first randomly sample videos from the WebVid [2] dataset, which only contains video frames and lacks camera annotations. Then, we annotate the camera poses in each video clip using Particle-SfM [61]. Particle-SfM is a pre-trained framework capable of estimating camera trajectories for monocular videos captured in dynamic scenes. Due to the time-consuming nature of this process, we randomly sample 32 frames from each original video to feed into the Particle-SfM [61] pipeline.

Curating High-quality Samples

Note that the WebVid dataset contains videos of various kinds, including many with static cameras. To ensure the model learns to handle more challenging motions, we need to
filter out sequences where the estimated camera trajectories show minimal displacement.
Due to the inevitable ambiguity between object motion and camera motion, in-the-wild estimation of camera poses can be inaccurate.
We thus use the number of points in the reconstructed sparse point clouds from Particle-SfM [61] as an indicator of camera annotation quality.
The intuition is that an accurate camera pose estimation comes from well-registered frames. If the frames have more 3D-consistent pixels, more point correspondences will contribute to solving the camera parameters.
With this indicator, we only use videos with accurate camera pose estimation and high object motion,
and construct a training set of high-quality 12,000 video sequences annotated with camera poses.

We mainly compare our method against Stable Video Diffusion [1], VideoCrafter [8], and MotionCtrl [51].
For MotionCtrl [51], we take the image-to-video generation checkpoint released by the authors for a fair comparison since their released checkpoint is also trained from Stable Video Diffusion [1].
For each sequence in RealEstate10k [62], we first randomly select a starting frame and then take every 8th frame to ensure reasonable camera change.
Note that in our evaluations and visualizations, none of the input images or camera trajectories was seen during training.

Since our major goal is to generate videos that follow the camera motion input, we need to measure the camera pose from the output videos. Thanks to COLMAP [40, 41], we are able to annotate the camera poses of videos we obtain. Because structure-from-motion algorithms can only estimate the scene structure up to a certain scale, we further canonicalize the estimated camera poses before calculating the differences. Specifically, we follow our training protocol where we first convert the camera systems to be relative to the first frame and then normalize the scale of the scene by finding the furthest camera against the first frame.

COLMAP error rate and matched points

Due to the randomness introduced in COLMAP’s pipeline, we report the average rate with five trials for each video.
We consider a reconstruction to be successful only when a sparse model is created with all 14 frames matched together.
We further report the number of points available in the reconstructed sparse point clouds as a reflection of the 3D consistency of the video frames.

We extract the camera-to-world matrices of the COLMAP predictions and obtain the estimated rotation and translation vectors, similar to CameraCtrl [15].
The relative rotation distances are then converted to radians, and we sum the total error of 14 frames,



Rerr=∑i=1narcos​(tr​(RoutiT​Rgti)−12).R_{\text{err}}=\sum_{i=1}^{n}\text{arcos}(\frac{\text{tr}(R_{\text{out}_{i}}^{T}R_{\text{gt}_{i}})-1}{2}).

(4)

The norm of the relative translation vector for each frame is also summed together to form the translation error of the whole video,



Terr=∑i=1n‖Touti−Tgti‖2.T_{\text{err}}=\sum_{i=1}^{n}\left\|T_{\text{out}_{i}}-T_{\text{gt}_{i}}\right\|_{2}.

(5)

Additionally, we evaluate the visual quality of the generated video frames, regardless of the camera following ability or geometry quality. FID [17] and FVD [48] are calculated based on deep features extracted from video frames. We, therefore, use FID [17] and FVD [48] to measure the distance between the generated frames and the corresponding reference videos.

Due to the lack of large-scale camera pose annotation of dynamic videos in the wild, we quantitatively evaluate our method against previous works on static videos. Specifically, we randomly sampled 1,000 videos from the Realestate-10k [62] test set. The camera poses were originally annotated using COLMAP [40, 41]. We provide qualitative and quantitative comparisons in Fig. 3 and Tab. 1, respectively. In Fig. 3, the ground truth cameras are visualized through the camera reference videos. Compared to the baselines, our method produces results with the best camera-following ability. In contrast, the results produced by Stable Video Diffusion do not follow the camera control, and MotionCtrl demonstrates inaccurate camera-following results. Our results outperform the baselines by a large margin in all metrics, indicating our superiority in visual quality, camera controllability, and geometric consistency. Note that for all experiments, neither the input image nor the camera trajectory was seen during training.

Similar to the static scenario, we evaluate our method against previous works on randomly sampled 1,000 videos from the test split of our annotated WebVid [2] dataset. We provide qualitative and quantitative comparisons in Fig. 4 and Tab. 3, respectively. In Fig. 4, we provide the generated videos when inferencing text-to-image results produced by SDXL [34]. The last column provides the camera trajectories. Compared to the baselines, our method excels at producing perspective changes as well as object motion. Notably, in the eagle case, the baselines fail to produce reasonable object motion or follow the camera change configurations, whereas our method successfully generates both camera changes and object movements vividly. In Tab. 4, we provide FID and FVD calculated against ground truth videos sampled from the WebVid dataset. Our full model produces the best FID and FVD values, indicating the model produces the best visual quality. In comparison, the model trained on the uncurated dynamic dataset and the static version model show inferior results, indicating poorer object motion.
Note that for all experiments, neither the input image nor the camera trajectory was seen during training.

In this section, we conduct several ablation studies on the model variants, including different versions of camera conditioning and the importance of our epipolar attention block. “w/o epipolar attention” denotes the baseline where we remove the epipolar constraint attention from our model and only use the temporal convolution layers. “w/o Plücker” refers to the model variant where we remove the Plücker coordinate input in the temporal blocks and rely on the epipolar constraint attention to modulate the camera control. “1-dim camera” means using 1-dimensional camera matrices as the conditioning signal for modulating the features in temporal blocks instead of using 2-dimensional Plücker coordinates. “Time embedding” refers to modulating the time embedding feature with camera matrices instead of using Plücker coordinates. As can be seen in Tab. 2, the full model delievers best results in both 2D visual quality (FID and FVD) and camera pose accuracy (COLMAP error, points, translation and rotation error). In contrast, the model variants lack the ability of accurately controlling the camera poses and can lead to deteriorated visual appearance.

In this paper, we introduced CamCo, a camera-controllable framework that produces 3D-consistent videos. Our method is built upon a pre-trained image-to-video diffusion model and introduces novel camera conditioning and geometry constraint blocks to ensure camera control accuracy and geometry consistency. Further, we proposed a data curation pipeline to improve the generation of object motion. Our experiments on images of various domains demonstrate the superiority of CamCo against previous works in terms of camera controllability, geometry consistency, and visual quality.

Appendix A Additional Details on Epipolar Constraint Attention

An epipolar line refers to the projection on one camera’s image plane of the line connecting a point in 3D space with the center of projection of another camera. Specifically, let 𝐊,𝐑,𝐓\mathbf{K},\bf{R},\bf{T} be the relative camera pose between frame ii and the first frame,
the projections of point pp and camera origin o=(000)o=(\begin{smallmatrix}0\\
0\\
0\end{smallmatrix}) on source view the first frame are denoted as Proj​(𝐑⁡(𝐊−1​𝐩i)+𝐓)\text{Proj}\left(\mathbf{R}\left(\mathbf{K}^{-1}\mathbf{p}^{i}\right)+\mathbf{T}\right) and Proj​(𝐑⁡(𝐊−1​(000))+𝐭)\text{Proj}\left(\mathbf{R}\left(\mathbf{K}^{-1}(\begin{smallmatrix}0\\
0\\
0\end{smallmatrix})\right)+\mathbf{t}\right), respectively, where Proj is the projection function. The epipolar line is, therefore, given by



𝐋=𝐨+c⁡(𝐩−𝐨),\mathbf{L}=\mathbf{o}+c\left(\mathbf{p}-\mathbf{o}\right),

(6)

where c∈{0,∞}∈ℝc\in\{0,\infty\}\in\mathbb{R}.

Appendix B Experiment Details

B.1 COLMAP configuration

We use the widely adopted COLMAP configuration for few-view 3D reconstruction [11] and assume all frames in each video share the same camera intrinsics. For the feature extractor, we set
--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --ImageReader.single_camera 1 . For the feature matching process, we set --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 65536. For each video, we retry the process five times, at most.

B.2 Particle-sfm workflow

We use the full default configuration of Particle-sfm for the best performance. Videos are processed at their original resolution without any resizing or cropping operation. Given a long video sequence, we randomly sample a frame stride and then uniformly obtain a set of 32 frames according to the stride. The 32 frames are then sent into Particle-sfm for camera annotation.

B.3 Additional Settings

We use the same train-test split of RealEstate-10k [62] as in PixelSplat [7].
Baseline methods are ran at their originally designed resolution. Thanks to the image-to-video nature, when the input image is at the resolution of 256×256256\times 256, despite what the output image resolution is, the output becomes undistorted when being resized to 256×256256\times 256. For baselines, we use the open-source checkpoints released by the authors. We use ‘‘clean-fid’’11
1
https://github.com/GaParmar/clean-fid and ‘‘common-metrics-on-video-quality’’22
2
https://github.com/JunyaoHu/common_metrics_on_video_quality for calculating FID and FVD, respectively. For FVD, we report the results in VideoGPT [55] format.

Appendix C Implementation Details

Our CamCo model builds upon SVD [1], which is a UNet-based open-source image-to-video diffusion model with architecture similar to VideoLDM [4].
In our experiments, we use a relative camera system where all camera poses are converted to become relative to the first frame. The camera of the first frame is placed to the world origin and rotated to the face towards the x-axis. During training, we randomly sample the strides of the image sequences to augment the speed of the video. The sampled camera extrinsics are then normalized to have a unit maximum distance against the world origin to stabilize the training.
For simplicity and efficiency, the training frames are first center-cropped to square images and then downsampled to 256×256256\times 256 resolution. During inference, we use 25 sampling steps to obtain the 14 frames. The decoding chunk size of the latent decoder is set to 14 frames for the best visual quality. Our model is trained with batch-size 64 in total on 16 A100 GPUs for around two days. The learning rate is set to 2e-5 with a warmup of 1,000 steps using the Adam optimizer. We use fp16 training implemented with ‘‘accelerate’’33
3
https://github.com/huggingface/accelerate library.

Appendix D Limitations and Future Work

Because our training data are video frames with the same camera intrinsics, our model generates images that have the same camera intrinsic as the input image. Therefore, the model can not generate complex camera intrinsic changes such as dolly zoom. Additionally, CamCo is able to generate 14 frames at the resolution of 256×256256\times 256, which only covers limited viewpoints and can be insufficient for large scenes. We will explore generating longer, and larger resolution videos in future work.

Appendix E Broad Impacts

The paper focuses on generating videos from image input. Like all other large models trained from large-scale datasets, our model can contain certain social biases. Such biases could perpetuate stereotypes or misrepresentations in generated videos, thus influencing public perceptions and potentially leading to societal harm.
Our T2I base model is trained with NSFW content removed and is also released with a safety checker. The publicly available safety checker maintained by the diffusers library can also protect our video generator from harmful usage.

Instructions for reporting errors