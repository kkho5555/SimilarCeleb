// //# import * as tf from '@tensorflow/tfjs';
function opencvIsReady() {
  console.log('ready');
}

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $('.image-upload-wrap').hide();
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
      let utils = new Utils('errorMessage');
      utils.loadImageToCanvas(e.target.result, 'canvasInput');
      utils.loadOpenCv(() => {
        let faceCascadeFile = 'haarcascade_frontalface_alt.xml';
        utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
          // tryIt.removeAttribute('disabled');
          faceDetect();
        });
      });
    };

    reader.readAsDataURL(input.files[0]);
  } else {
    alert('img');
  }
}
function faceDetect() {
  let src = cv.imread('canvasInput');
  let gray = new cv.Mat();
  let resultImg = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
  let faces = new cv.RectVector();
  let faceCascade = new cv.CascadeClassifier();
  // load pre-trained classifiers
  console.log(faceCascade.load('haarcascade_frontalface_alt.xml'));

  // detect faces
  let msize = new cv.Size(0, 0);
  faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
  let dsize = new cv.Size(64, 64);
  // You can try more different parameters

  src = gray.roi(faces.get(0));
  cv.resize(src, src, dsize, 0, 0, cv.INTER_AREA);
  // cv.crop(src, point1, point2, [255, 0, 0, 255]);
  // }
  cv.imshow('canvasOutput', src);
  src.delete();
  gray.delete();
  faceCascade.delete();
  faces.delete();
}

// async function init() {

//   const model = await tf.loadLayersModel('../celeb/tfjs_artifacts/model.json');
//   const img = document.getElementById('_img');
//   const tfImg = tf.browser.fromPixels(img);
//   const smalImg = tf.image.resizeBilinear(tfImg, [64, 64]);
//   const resized = tf.cast(smalImg, 'float32');
//   const t4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 64, 64, 3]);

//   const labelContainer = document.getElementById('label-container');
//   for (let i = 0; i < 10; i++) {
//     // and class labels
//     labelContainer.appendChild(document.createElement('div'));
//   }
//   //const example = tf.fromPixels(img); // for example
//   const prediction = await model.predict(t4d, true);
//   console.log(model.summary());
//   const divv = document.getElementById('_div');
//   console.log();
//   divv.innerHTML = prediction.dataSync();
// }
// init();
