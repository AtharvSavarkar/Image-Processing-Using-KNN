# Image-Processing-Using-KNN (KNN-Smoothing)
Image Processing using KNN (K Nearest Neighbours Algorithim)




## knn() function takes 5 arguments (4 of them are optional)

### 1. image_path (string) - This should contain path to the image with it's extension
- e.g. image_path = 'original_images/image1.jpg'
- All types of images (jpg, png, etc) are supported as long as they can be read as matrices (multi dimensional)
### 2. k (int or float representation of integer) - This is the number of neighbours to consider while performing smoothing
   - k can vary from 0 to (n^2 - 1) - where n is square matrix side taken under consideration
   - k = 0 implies that only center pixel will be considered (no smoothing)
### 3. n (int or float representation of integer) - This is matrix size used to implement KNN smoothing
- n can be as large as min(length, breadth) of image
- n should be odd as center pixel needs to be located for finding KNN w.r.t center pixel
### 4. print_logs (bool True or False) (default -> True)
   - If print_logs == True then logs for the code will be printed. It will contain image name, size, time to compute, etc
   - If print_logs == False then no logs will be printed
### 5. save_image (bool True or False) (default -> True)
- If True image will be saved in 'processed_image' folder (if folder is not present it will be created)
- If False computed image will not be saved
- Saved image name will contain information regarding n and k used
- Image saved will be .jpg format (this can be changed by changing extension in knn() function)
### 6. return_image_frame (bool True or False) (default -> False)

- If True knn() will return image frame (3D matrix)
- b. If False function will not return anything (None)
