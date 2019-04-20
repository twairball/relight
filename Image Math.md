# Image Math

Gamma correction:  A * ((x / A) ** 1/gamma)

Brightness:  (R+G+B) / 3

Contrast: (I_max - I_min) / (I_max + I_min)

Adjust contrast/brightness:
		g = a * x + b   ; where a = contrast, b = brightness

Luminance:  L = 0.299 * X_r + 0.587 * X_g + 0.114 * X_b

