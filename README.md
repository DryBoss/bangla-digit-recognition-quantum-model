## Image Preprocessing and Flattening

This section describes the preprocessing pipeline applied to the dataset images before using them in machine learning models.

### Steps:

1. **Load Images**  
   - Images are loaded from the `grayscale` and `binary` folders.
   - Each image is converted to **grayscale**.

2. **Crop Whitespace**  
   - Extra background pixels are removed using a threshold.
   - Only the region containing the actual digit/object is retained.

3. **Resize Images**  
   - Cropped images are resized to a fixed size (default `16x16`) suitable for ML input.

4. **Normalize**  
   - Pixel values are scaled to `[0,1]`.

5. **Optional Binarization**  
   - Images can be converted to binary using a threshold of `0.5`.

6. **Flatten**  
   - Each processed image is flattened into a 1D array for model training.

7. **Save Arrays**  
   - Processed arrays and corresponding labels are saved as `.npy` files in `features/training-a/flattened_arrays`.

### Output

- `X_gray.npy` and `y_gray.npy` – flattened grayscale images and labels.
- `X_binary.npy` and `y_binary.npy` – flattened binary images and labels.

This pipeline ensures uniform, normalized, and ready-to-use image data for machine learning models.