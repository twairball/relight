# Relight

Fixing under-exposed photos when taken outdoors, with the sun in the background.

- Histogram Equalization (HE)
- Contrast Limited Adapative HE (CLAHE)
- Dynamic HE (DHE)
- Adapative Local Tone Mapping (ALTM)

# Requirements

python >= 3.6.0
numpy >= 1.15
opencv >= 3.4.1

# Usage

````
python main.py --src_dir=temp/jpg/ --target_dir=temp/out/
````