# SimilarCeleb
닮은 연예인찾기 프로젝트입니다.


1. 이미지 수집,얼굴인식, 그레이스케일
   1) Naver Image Crawler (/model/Image_Crawling.ipynb)
   2) OpenCV - Grayscaling and face detecting

2. 이미지 전처리(model/celeb/pre_processing.ipynb)
   1) Looking at the images one by one, delete irrelevant data.
   2) Preprocessing of image data using numpy serializing to save out file(.npy).


3. 모델 학습(model/celeb/celeb_classification.ipynb)
   1) Load npy file and separate it into train set and test set.
   2) After the keras model declaration, model learning is carried out by changing the values of each parameter.
   3) After learning is finished, the saved model file is converted using the "tfjs" package.

4. 시각화(process.js)
   1) When the user inputs an image, face detection and gray scaling are performed using OpenCvJS.
   2) Match the input shape of the trained model with the input shape of the image entered by the user.
   3) Predict the preprocessed image.
   4) Visualize the results.
