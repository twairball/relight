import cv2
import numpy as np

def convolve(src, kernel):
    knl = cv2.flip(kernel, 0)
    return cv2.filter2D(src, -1, knl, borderType=cv2.BORDER_REPLICATE)

def covariance(x, y, shape=(3,3)):
    """Covariance filter"""

    mu_x = cv2.blur(x, shape)
    mu_y = cv2.blur(y, shape)
    xy = cv2.multiply(x, y)
    mu_xy = cv2.blur(xy, shape)

    cov = mu_xy - cv2.multiply(mu_x, mu_y)
    return cov

def pearson(x, y, shape=(3,3)):
    """Pearson Correlation Coeff filter"""
    mu_x = cv2.blur(x, shape)
    mu_y = cv2.blur(y, shape)
    xy = cv2.multiply(x, y)
    mu_xy = cv2.blur(xy, shape)

    cov = mu_xy - cv2.multiply(mu_x, mu_y)

    mu_x2 = cv2.blur(cv2.multiply(x, x), shape)
    mu_y2 = cv2.blur(cv2.multiply(y, y), shape)
    
    var_x = mu_x2 - cv2.multiply(mu_x, mu_x)
    var_x = var_x.astype(np.float64)
    sigma_x = cv2.sqrt(var_x)

    var_y = mu_y2 - cv2.multiply(mu_y, mu_y)
    var_y = var_y.astype(np.float64)
    sigma_y = cv2.sqrt(var_y)

    rho = cov / (cv2.multiply(sigma_x, sigma_y))
    return rho 


def build_is_hist(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    fv = fh.conj().T
    
    [H, S, I] = cv2.split(hsv)

    dIh = convolve(I, np.rot90(fh, 2))
    dIv = convolve(I, np.rot90(fv, 2))
    dIh[dIh==0] = 0.00001
    dIv[dIv==0] = 0.00001
    di = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
    
    dSh = convolve(S, np.rot90(fh, 2))
    dSv = convolve(S, np.rot90(fv, 2))
    dSh[dSh==0] = 0.00001
    dSv[dSv==0] = 0.00001
    ds = cv2.sqrt(cv2.pow(dSh, 2) + cv2.pow(dSv, 2)).astype(np.uint32)

    Imean = convolve(I, np.ones((5,5))/25.0)
    Smean = convolve(S, np.ones((5,5))/25.0)
    
    print("building Rho corrcoefs")
    rho = pearson(I, S, shape=(3,3))
    rho[np.isnan(rho)] = 0
    rd = (rho*ds).astype(np.uint32)
    Hist_I = np.zeros((256,1))
    Hist_S = np.zeros((256,1))
    
    # TODO: needs optimizing 
    print("building histograms...")
    for n in range(0,255):
        temp = np.zeros(di.shape)
        temp[I==n] = di[I==n]
        Hist_I[n+1] = np.sum(temp.flatten('F'))
        temp = np.zeros(di.shape)
        temp[I==n] = rd[I==n]
        Hist_S[n+1] = np.sum(temp.flatten('F'))

    return Hist_I, Hist_S


def output(img, hist_i, hist_s, alpha=0.5):
    
    [h, s, i] = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    hist_c = alpha*hist_s + (1-alpha)*hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)

    s_r = hist_cum / hist_sum
    i_s = np.zeros(i.shape)
    for n in range(0,255):
        i_s[i==n] = s_r[n+1]
    i_s[i==255] = 1
    i_s = (i_s * 255).astype(np.uint8)

    hsi_o = cv2.merge([h,s,i_s])
    result = cv2.cvtColor(hsi_o, cv2.COLOR_HSV2BGR)
    return result 

def dhe(img, alpha=0.5):

    # build histogram on smaller sample
    img_sm = cv2.resize(img, (300,300))
    hist_i, hist_s = build_is_hist(img_sm)

    # make output image
    result = output(img, hist_i, hist_s)
    return result 
