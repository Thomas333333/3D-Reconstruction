import argparse
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
import argparse
import open3d as o3d
import random


def read_img(img_path):
    # 读取图片地址，转化为灰度图
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img = img
    img.astype('float32')
    return img


def extract_features(image):
    """
    提取图像特征
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    # print(kp[0].pt[0],kp[0].pt[1],kp[0].size)
    kp2array = np.array([(k.pt[0], k.pt[1], k.size) for k in kp])

    # 检查是否修改了匹配关系
    return kp2array, des


def match_keypoints(kp1, des1, kp2, des2):
    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = np.array([(m.queryIdx, m.trainIdx) for m in matches])
    return matches


# 使用Lowe's ratio test, 特征点变少，生成的三维重建特征点变少，从6700到500
# def match_keypoints(kp1,des1,kp2,des2):
#     matcher = cv2.BFMatcher_create()
#     knn_matches = matcher.knnMatch(des1,des2,k=2)

#     matches = []
#     #使用 Lowe's ratio test
#     for m, n in knn_matches:
#         if m.distance < 0.7 * n.distance:
#             matches.append(m)
#     matches = np.array([(m.queryIdx,m.trainIdx) for m in matches])
#     return matches

def GetPairs(kp1, des1, kp2, des2):
    # 使用暴力搜索进行特征点匹配
    matches = match_keypoints(kp1, des1, kp2, des2)
    data1 = []
    data2 = []
    for m in matches:
        pt1 = [kp1[m[0]][0], kp1[m[0]][1]]
        pt2 = [kp2[m[1]][0], kp2[m[1]][1]]
        data1.append(pt1)
        data2.append(pt2)

    return data1, data2


def line2pics(img1, img2, kp1, des1, kp2, des2):
    # 连线
    matches = match_keypoints(kp1, des1, kp2, des2)
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)

    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1[:, :, i]
        newimg[:h2, w1:w1 + w2, i] = img2[:, :, i]

    for m in matches:
        #     print(coordinate2[img2_index])
        pt1 = (int(kp1[m[0]][0]), int(kp1[m[0]][1] + hdif))
        pt2 = (int(kp2[m[1]][0] + w1), int(kp2[m[1]][1]))
        # print(pt1,pt2)
        ran_bgr = np.random.randint(0, 255, 3, dtype=np.int32)
        cv2.line(newimg, pt1, pt2, (int(ran_bgr[0]), int(ran_bgr[1]), int(ran_bgr[2])), 2)
        cv2.circle(newimg, pt1, 1, (255, 0, 0), 15)
        cv2.circle(newimg, pt2, 1, (0, 255, 0), 15)
    #     print(pt1,pt2)
    plt.subplot(2, 1, 1)
    plt.imshow(newimg)
    plt.title("SIFT points pairs")



def path2pairs(image1_path, image2_path):
    # 读取灰度图
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    gray1 = read_img(image1_path)
    gray2 = read_img(image2_path)
    # 程序参数读写无误
    # plt.imshow(img1)
    # plt.show()
    # 利用SIFT算子进行特征点识别
    kp1, des1 = extract_features(gray1)
    kp2, des2 = extract_features(gray2)
    # 输出两张图片的匹配结果
    # line2pics(img1, img2, kp1, des1, kp2, des2)
    # 进行特征点匹配，输出匹配数据
    data1, data2 = GetPairs(kp1, des1, kp2, des2)

    data1 = np.array(data1)
    data2 = np.array(data2)

    return data1, data2


def get_intrinsic_matrix(intrinsics_path):
    """
    读取相机内参信息,格式为fx fy ox oy
    """
    with open(intrinsics_path, 'r') as f:
        # K = np.array([[float(x) for x in line.split()] for line in f])
        K_info = []
        for line in f:

            if line[0] == '#':
                continue
            if "," in line:
                for x in line.strip().split(','):
                    K_info.append(float(x))
            else:
                for x in line.strip().split(' '):
                    K_info.append(float(x))
        K = np.array([[K_info[0], 0, K_info[2]], [0, K_info[1], K_info[3]], [0, 0, 1]])

    return K


def construct_A(data1n, data2n):
    # 归一化后的匹配点的数据  用于八点法中构造A矩阵
    x1, y1 = data1n[:, 0], data1n[:, 1]
    x2, y2 = data2n[:, 0], data2n[:, 1]
    # print(x1)
    # print(y1)
    N = len(x1)
    A = np.array([
        x1 * x2,
        x2 * y1,
        x2,
        x1 * y2,
        y1 * y2,
        y2,
        x1,
        y1,
        np.ones(N)
    ])
    A = np.stack(A, axis=1)  # (9, 295)变(295, 9)
    return A


def GetE(data1n, data2n, norm=False):
    # 八点法求解E矩阵
    A = construct_A(data1n, data2n)
    U, S, V = np.linalg.svd(A[:8])  # 295*295 295*9 9*9
    E_est = V[-1]
    E_est = E_est.reshape(3, 3)

    if norm:
        E_est = E_est / E_est[2, 2]

    return E_est


def FinalGetE(data1n, data2n, norm=False):
    # 在原先求解E矩阵的时候加入修正保持其代数性质
    A = construct_A(data1n, data2n)
    U, S, V = np.linalg.svd(A[:8])  # 295*295 295*9 9*9
    E_est = V[-1]
    E_est = E_est.reshape(3, 3)
    U, S, V = np.linalg.svd(E_est)
    S_new = [(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0]
    S_new = np.array(S_new)
    S_new = np.diag(S_new)
    E_est = U @ S_new @ V

    if norm:
        E_est = E_est / E_est[2, 2]

    return E_est


def ComputeRT(E):
    # 从本质矩阵恢复R和T
    U, _, VT = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(VT) < 0:
        VT[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    WT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    R1 = np.dot(U, np.dot(W, VT))
    R2 = np.dot(U, np.dot(WT, VT))

    T1 = U[:, 2]
    T2 = -U[:, 2]

    return R1, R2, T1, T2


def triangulate_dlt(R, T, points1, points2):
    # 根据R、T恢复三维坐标点
    assert points1.shape == points2.shape, "Input point arrays must have the same shape"
    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=-1)
    # 得到3x4矩阵
    P2 = np.concatenate((R, T.reshape(3, 1)), axis=-1)
    # 得到3x4矩阵
    num_points = points1.shape[0]
    points_3d = np.zeros((num_points, 3))
    for i in range(num_points):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        A = np.array([
            [x1 * P1[2] - P1[0]],
            [y1 * P1[2] - P1[1]],
            [x2 * P2[2] - P2[0]],
            [y2 * P2[2] - P2[1]]
        ]).reshape(4, 4)
        _, _, V = np.linalg.svd(A)
        X_homogeneous = V[-1]
        X = X_homogeneous[:3] / X_homogeneous[3]  # 标准化
        points_3d[i] = X

    return points_3d


def JudgeFinalEst(E, data1n, data2n):
    R1, R2, T1, T2 = ComputeRT(E)
    # 由R1、R2、T1、T2可构成四种组合，一一验证其合理性
    est1_3d = triangulate_dlt(R1, T1, data1n, data2n)
    est2_3d = triangulate_dlt(R1, T2, data1n, data2n)
    est3_3d = triangulate_dlt(R2, T1, data1n, data2n)
    est4_3d = triangulate_dlt(R2, T2, data1n, data2n)
    est_array = [est1_3d, est2_3d, est3_3d, est4_3d]
    flag = 0
    for item in est_array:
        if (item[:, 2] >= 0).all():
            best = item
            flag = 1
    if flag == 0:
        # raise Exception("No match RT")
        return None,0
    else:
        final_3dest = best

    return final_3dest,flag


def Reconstruct3D(data1, data2, K):
    data1h = np.concatenate((data1, np.ones((data1.shape[0], 1))), axis=-1)
    data2h = np.concatenate((data2, np.ones((data2.shape[0], 1))), axis=-1)
    # print(data1n)
    # 利用相机内参归一化坐标
    data1n = (np.linalg.inv(K) @ (data1h.T)).T
    data2n = (np.linalg.inv(K) @ (data2h.T)).T

    print(f'SIFT特征子匹配点对数{data1n.shape[0]}')
    # 利用八点法计算本质矩阵E
    flag = 0
    max_interation = 10
    interation =1
    while flag == 0:
        print(f'RANSAC times :{interation}')
        if interation ==max_interation:
            raise Exception("After cycling so many times we can't find match RT")

        E, best_pairs = RANSAC(data1n, data2n)
        #     # print(best_pairs)
        #     # best_pairs=range(500)

        data1n_new = data1n[best_pairs]
        data2n_new = data2n[best_pairs]


        # 从本质矩阵E中分解出R和T,并根据不同的RT组合生成不同的三维点
        # 最后根据Z的正负选出最合适的RT。
        final_est_3d ,flag= JudgeFinalEst(E, data1n_new, data2n_new)
        interation+=1
    print(f'RANSAC算法后剩余匹配点对数{data1n_new.shape[0]}')
    pcd_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_est_3d))
    o3d.visualization.draw_geometries([pcd_est])
    return (K @ data1n_new.T).T, (K @ data2n_new.T).T


def matchpics(img1, img2, data1, data2):
    # 连线

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)

    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1[:, :, i]
        newimg[:h2, w1:w1 + w2, i] = img2[:, :, i]

    for i in range(len(data1)):
        #     print(coordinate2[img2_index])
        pt1 = (int(data1[i, 0]), int(data1[i, 1] + hdif))
        pt2 = (int(data2[i, 0] + w1), int(data2[i, 1]))
        # print(pt1,pt2)
        ran_bgr = np.random.randint(0, 255, 3, dtype=np.int32)
        cv2.line(newimg, pt1, pt2, (int(ran_bgr[0]), int(ran_bgr[1]), int(ran_bgr[2])), 2)
        cv2.circle(newimg, pt1, 1, (255, 0, 0), 15)
        cv2.circle(newimg, pt2, 1, (0, 255, 0), 15)
    #     print(pt1,pt2)
    plt.subplot(2, 1, 2)
    plt.imshow(newimg)
    plt.title('After RANSAC')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def RANSAC(points1, points2, confidence=0.99, threshold=0.02, max_interation=1000, outline_rate=0.7):
    # 使用RANSAC方法找到内点和合适的E矩阵
    assert points1.shape == points2.shape, "Input point arrays must have the same shape"
    num_points = points1.shape[0]
    best_inliers = []
    best_E = None

    interation = max_interation
    if 0 < confidence < 1:
        interation = int(np.log(1 - confidence) / np.log(1 - (1 - outline_rate) ** 8))
    else:
        raise Exception("Confidence Value Error")
    # print(interation)
    # array =0
    for i in range(interation):
        inliers = []
        # 随机采样
        # print(i)
        indices = random.sample(range(num_points), 8)
        sample1 = points1[indices]
        sample2 = points2[indices]

        ###################################是否加入修正

        E = GetE(sample1, sample2)

        # compute the epipolar lines

        line1 = np.dot(E.T, points1.T).T
        line2 = np.dot(E, points2.T).T
        # 不太理解这里
        distance1 = np.abs(np.sum(line1 * points2, axis=1)) / np.sqrt(line1[:, 0] ** 2 + line1[:, 1] ** 2)
        distance2 = np.abs(np.sum(line2 * points1, axis=1)) / np.sqrt(line2[:, 0] ** 2 + line2[:, 1] ** 2)

        #         distance1 = np.abs(np.sum(1 * points1, axis=1))
        #         distance2 = np.abs(np.sum(2 * points1, axis=1))

        total_distance = distance1 + distance2

        inliers = np.where(total_distance < threshold)[0]
        # inliers = range(400)
        # array += len(inliers)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_E = E
        # break
    # print(len(best_inliers))
    # print(array/interation)
    return best_E, best_inliers

if __name__ == '__main__':
    #获取参数信息
    parser = argparse.ArgumentParser(description='Image matching and 3D reconstruction')
    parser.add_argument('-image1', type=str, help='path to first image', required=True)
    parser.add_argument('-image2', type=str, help='path to second image', required=True)
    parser.add_argument('-intrinsics', type=str, help='path to camera intrinsic parameters', required=True)
    args = parser.parse_args()
    # print(args)#测试无问题

    image1_path = args.image1
    image2_path = args.image2
    intrinsics_path = args.intrinsics
    data1,data2 = path2pairs(image1_path, image2_path)
    '''设计读取相机内参K的函数'''
    K = get_intrinsic_matrix(intrinsics_path)
    # print(K)


    data3, data4 = Reconstruct3D(data1, data2, K)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    matchpics(img1, img2, data3, data4)#输出匹配图


