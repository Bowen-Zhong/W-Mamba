import numpy as np
from scipy.signal import convolve2d
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def EN_function(image_array):
    # 计算图像的直方图
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def SD_function(image_array):
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD

def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR

def MSE_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE

def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    return vifp

def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF

def CC_function(A, B, F):
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(
        np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(
        np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC

def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r

def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r

def Hab(im1, im2, gray_level):
    hang, lie = im1.shape
    count = hang * lie
    N = gray_level
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    h = h / np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    H_x = 0
    H_y = 0
    for i in range(N):
        if im1_marg[i] != 0:
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
    for i in range(N):
        if im2_marg[i] != 0:
            H_y = H_y + im2_marg[i] * math.log2(im2_marg[i])
    H_xy = 0
    for i in range(N):
        for j in range(N):
            if h[i, j] != 0:
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])
    MI = H_xy - H_x - H_y
    return MI

def MI_function(A, B, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI_results = MIA + MIB
    return MI_results

def MI_function2(A, B, C, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MIC = Hab(C, F, gray_level)
    MI_results = MIA + MIB + MIC
    return MI_results

def AG_function(image):
    width = image.shape[0]
    width = width - 1
    height = image.shape[1]
    height = height - 1
    tmp = 0.0
    [grady, gradx] = np.gradient(image)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG

def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = ssim_B/2 + ssim_A/2
    return SSIM.item()

def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = ssim_B/2 + ssim_A/2
    return MS_SSIM.item()

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            perms = list(range(win.ndim))
            perms[2 + i] = perms[-1]
            perms[-1] = 2 + i
            out = conv(out, weight=win.permute(perms), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.type_as(X)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs

def ssim(X,
         Y,
         data_range=255,
         size_average=True,
         win_size=11,
         win_sigma=1.5,
         win=None,
         K=(0.01, 0.03),
         nonnegative_ssim=False):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = torch.squeeze(X, dim=d)
        Y = torch.squeeze(Y, dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = F.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(dim=1)


def ms_ssim(
        X,
        Y,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        weights=None,
        K=(0.01, 0.03)
):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(dim=1)

class SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        ).item()


class MS_SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
    ):
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        ).item()


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def convolution(k, data):
    k = flip180(k)
    data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    n, m = data.shape
    img_new = []
    for i in range(n - 2):
        line = []
        for j in range(m - 2):
            a = data[i:i + 3, j:j + 3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)

def getArray(img):
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    SAx = convolution(h3, img)
    SAy = convolution(h1, img)
    gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
    n, m = img.shape
    aA = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (SAx[i, j] == 0):
                aA[i, j] = math.pi / 2
            else:
                aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
    return gA, aA

def getQabf(aA, gA, aF, gF, Tg = 0.9994, kg = -15, Dg = 0.5, Ta = 0.9879, ka = -22, Da = 0.8):
    n, m = aA.shape
    GAF = np.zeros((n, m))
    AAF = np.zeros((n, m))
    QgAF = np.zeros((n, m))
    QaAF = np.zeros((n, m))
    QAF = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (gA[i, j] > gF[i, j]):
                GAF[i, j] = gF[i, j] / gA[i, j]
            elif (gA[i, j] == gF[i, j]):
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i, j] / gF[i, j]
            AAF[i, j] = 1 - np.abs(aA[i, j] - aF[i, j]) / (math.pi / 2)

            QgAF[i, j] = Tg / (1 + math.exp(kg * (GAF[i, j] - Dg)))
            QaAF[i, j] = Ta / (1 + math.exp(ka * (AAF[i, j] - Da)))

            QAF[i, j] = QgAF[i, j] * QaAF[i, j]

    return QAF

def Qabf_function(strA, strB, strF):
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno

    return output
