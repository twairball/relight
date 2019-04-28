import cv2
import numpy as np

def convolve(src, kernel):
    knl = cv2.flip(kernel, 0)
    return cv2.filter2D(src, -1, knl, borderType=cv2.BORDER_REPLICATE)

def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    
    Img = [np.pad(img[:,:,i], (2,2), 'edge') for i in range(img.shape[2])]
    Img = cv2.merge(Img)

    hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)
    fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    fv = fh.conj().T
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    I = hsv[:,:,2]

    dIh = convolve(I, np.rot90(fh, 2))
    dIv = convolve(I, np.rot90(fv, 2))
    dIh[dIh==0] = 0.00001
    dIv[dIv==0] = 0.00001
    dI = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
    di = dI[2:hei+2,2:wid+2]
    
    dSh = convolve(S, np.rot90(fh, 2))
    dSv = convolve(S, np.rot90(fv, 2))
    dSh[dSh==0] = 0.00001
    dSv[dSv==0] = 0.00001
    dS = np.sqrt(dSh**2+dSv**2).astype(np.uint32)
    ds = dS[2:hei+2,2:wid+2]

    
    h = H[2:hei+2,2:wid+2]
    s = S[2:hei+2,2:wid+2]
    i = I[2:hei+2,2:wid+2].astype(np.uint8)

    Imean = convolve(I, np.ones((5,5))/25.0)
    Smean = convolve(S, np.ones((5,5))/25.0)
    
    # TODO: needs optimizing
    print("building Rho corrcoefs")
    Rho = np.zeros((hei+4,wid+4))
    for p in range(2,hei+2):
        for q in range(2,wid+2):
            tmpi = I[p-2:p+3,q-2:q+3]
            tmps = S[p-2:p+3,q-2:q+3]
            corre = np.corrcoef(tmpi.flatten('F'),tmps.flatten('F'))
            Rho[p,q] = corre[0,1]
    
    rho = np.abs(Rho[2:hei+2,2:wid+2])
    rho[np.isnan(rho)] = 0
    rd = (rho*ds).astype(np.uint32)
    Hist_I = np.zeros((256,1))
    Hist_S = np.zeros((256,1))
    
    print("building histograms...")
    for n in range(0,255):
        temp = np.zeros(di.shape)
        temp[i==n] = di[i==n]
        Hist_I[n+1] = np.sum(temp.flatten('F'))
        temp = np.zeros(di.shape)
        temp[i==n] = rd[i==n]
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
