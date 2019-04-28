# Image Math

Gamma correction:  A * ((x / A) ** 1/gamma)

Brightness:  (R+G+B) / 3

Contrast: (I_max - I_min) / (I_max + I_min)

Adjust contrast/brightness:
		g = a * x + b   ; where a = contrast, b = brightness

Luminance:  L = 0.299 * X_r + 0.587 * X_g + 0.114 * X_b

Guided Filter q = f(p, I) where I = guide, p = image src. 

Mean filter = blur(x)

Standard Deviation sigma = sqrt(blur(x.x) - blur(x).blur(x))

Covariance cov(x,y) = blur(x.y) - blur(x).blur(y)

Correlation Coeffs pearson(x,y) = cov(x,y) / sigma(x) . sigma(y)

