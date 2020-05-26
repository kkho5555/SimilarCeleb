const IMAGE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const IMAGE_DATA = {
  '0': '가인',
  '1': '공찬',
  '2': '차은우',
  '3': '닉쿤',
  '4': '전지현',
  '5': '설현',
  '6': '아이유',
  '7': '옹성우',
  '8': '유승호',
  '9': '한예슬',
  '10': '규현',
  '11': '박서준',
  '12': '강다니엘',
  '13': '탑',
  '14': '태연',
  '15': '하니',
  '16': '김구라',
  '17': '백현',
  '18': '보아',
  '19': '뷔',
};

const IMAGE_SIZE = 64;
TOPK_PREDICTIONS = 20;
const predictionsElement = document.getElementById('predictions');
let loader__message = document.querySelector('#loader__message');
async function predict() {
  loader__message.innerHTML = '데이터베이스에서 닮은 얼굴을 찾습니다';
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
    const pred = model.predict(batched);

    return pred;
  });
  console.log('logits', await logits.data());
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  // Show the classes in the DOM.
  showResults(classes);
  showReturn();
}
function showReturn() {
  const start__btn = document.querySelector('.start__btn');
  start__btn.style.display = 'block';
  start__btn.animate([{ opacity: 0 }, { opacity: 1 }], 5000);
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
      className: IMAGE_DATA[topkIndices[i]],
      probability: topkValues[i],
    });
  }
  return topClassesAndProbs;
}

function showResults(classes) {
  console.log(classes);
  let probs = {};

  const predictionContainer = document.createElement('ul');
  predictionContainer.className = 'skills-bar-container';

  for (let i = 0; i < 5; i++) {
    const celeb = classes[i].className;
    const prob = (classes[i].probability * 100).toFixed(2) + '%';

    const row = document.createElement('li');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'progressbar-title';

    const classTitle = document.createElement('h3');
    classTitle.innerText = celeb;

    classElement.appendChild(classTitle);

    const probsElement = document.createElement('span');
    probsElement.id = celeb + '-percent';
    probsElement.className = 'percent';

    classElement.appendChild(probsElement);
    row.appendChild(classElement);
    const barContainer = document.createElement('div');
    barContainer.className = 'bar-container';
    const progressbar = document.createElement('progress');
    progressbar.id = 'progress-' + celeb;
    progressbar.className = 'progressbar ' + 'progressbar-' + i;
    progressbar.value = 0;
    progressbar.max = '100';

    barContainer.appendChild(progressbar);
    row.appendChild(barContainer);

    predictionContainer.appendChild(row);

    probs[celeb] = prob;
  }
  var multiply = 4;
  for (let [celeb, percent] of Object.entries(probs)) {
    var delay = 700;

    setTimeout(function () {
      document.getElementById(celeb + '-percent').innerHTML = percent;
      document.getElementById('progress-' + celeb).value = percent.replace(
        '%',
        ''
      );
    }, delay * multiply);

    multiply++;
  }

  predictionsElement.insertBefore(
    predictionContainer,
    predictionsElement.firstChild
  );
}

function readURL(input) {
  loader__message.innerHTML = '사진을 찾았습니다! 얼굴을 인식합니다.';
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
  loader__message.innerHTML = '얼굴 인식중.....';
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
  loader.style.display = 'flex';
}
function load_finish() {
  const loader = document.querySelector('#loader-wrapper');
  loader.style.display = 'none';
}
