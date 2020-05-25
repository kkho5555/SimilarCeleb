const IMAGE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const IMAGE_SIZE = 64;
TOPK_PREDICTIONS = 12;
const predictionsElement = document.getElementById('predictions');
async function predict() {
  const model = await tf.loadLayersModel('js/model/model.json');
  const imgElement = document.querySelector('#canvasOutput');
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(255);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.div(offset);
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    return model.predict(batched);
  });

  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGE_CLASSES[topkIndices[i]],
      probability: topkValues[i],
    });
  }
  return topClassesAndProbs;
}

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < 5; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = '이름' + classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText =
      '확률' + (classes[i].probability * 100).toFixed(2) + '%';
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
    predictionContainer,
    predictionsElement.firstChild
  );
}

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      loading();
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
  let msize = new cv.Size(0, 0);
  let dsize = new cv.Size(64, 64);
  // load pre-trained classifiers
  faceCascade.load('haarcascade_frontalface_alt.xml');
  //Call the time-consuming function

  faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

  // detect faces
  // You can try more different parameters
  try {
    src = gray.roi(faces.get(0));
  } catch (error) {
    alert('얼굴인식 실패;;;;');
    location.reload();
  }
  cv.resize(src, src, dsize, 0, 0, cv.INTER_AREA);
  // cv.crop(src, point1, point2, [255, 0, 0, 255]);
  // }
  cv.imshow('canvasOutput', src);

  predict().then(load_finish);
  src.delete();
  gray.delete();
  faceCascade.delete();
  faces.delete();
}
function loading() {
  const loader = document.querySelector('#loader-wrapper');
  loader.style.display = 'block';
}
function load_finish() {
  const loader = document.querySelector('#loader-wrapper');
  loader.style.display = 'none';
}
