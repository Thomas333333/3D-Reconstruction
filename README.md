## Homework2

### 文件路径

需要将pair1、pair2、pair3、pair4的文件夹和该文件夹中的程序放到同一文件夹下进行测试。

├─pair1

├─pair2
		├─pair3

├─pair4

├─main.py

└─FH.ipynb

### 

### 主要程序

#### main.py——双目三维重建包括RANSAC和八点法构造E矩阵及分解

1. `path2pairs()`函数将两张图片路径作为输入，输出对应的匹配点对
   + 首先读取图片信息，将其转化为灰度图
   + `extract_features()`利用SIFT算子提取两张图片特征点，返回坐标和特征向量
   + `line2pics()`输出两张图片的提取特征后的对应匹配结果
   + `GetPairs()`利用暴力搜索算法进行特征点匹配，输出匹配点对
2. `Reconstruct3D()` 函数将匹配点对和相机内参作为输入，输出三维重建的结果
   + 首先拓展坐标维度，填充1至三维
   + 利用相机内参归一化坐标
   + ~~`FinalGetE()`函数利用八点法计算本质矩阵E,最后修正了$\Sigma $~~
   + 若没有找到满足约束的RT，循环下列函数
     + 使用`RANSAC()`函数采用RANSAC算法，得到最佳的本质矩阵E和可行的内点的序号，
     + `JudgeFinalEst()`函数
       + `ComputeRT()`从E中分解出R和T
       + `triangulate_dlt()`根据不同的RT组合生成四个不同的三维点组
       + 通过判断三维点组的Z正负选择最合适的RT组合
   + 利用open3d可视化
3. `matchpics()`用于输出RANSAC后两张图片的匹配点对情况

#### FH.ipynb——八点法求解F矩阵、四点法求解F矩阵，并分解

1.`fromFGetRT(points1,points2,K)`*八点法求解F矩阵*

+ 将点进行归一化，并构建A矩阵
+ 将奇异值进行修正，令最后一位奇异值为0
+ 再利用`denormalize_matrix(F, pts1,pts2) `进行解归一化
+ `essential_from_fundamental(F, K)`将F矩阵转换为E矩阵
+ `ComputeRT(E)`利用之前的函数从E矩阵求解变换RT

2.`fromHGetE(pts1, pts2,K)`*四点法求解F矩阵*

+ `four_point_homography(pts1, pts2)`四点法求解H矩阵	
+ `homography_calibrated(K, H_uncalib)`将矩阵归一化
+ `get_motion_from_homography(H, K)`从H中获得变换矩阵



