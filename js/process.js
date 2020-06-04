const IMAGE_DATA = {
  "0": "아이비",
  "1": "아이린",
  "2": "안재현",
  "3": "안영미",
  "4": "비와이",
  "5": "찬성",
  "6": "치타",
  "7": "조세호",
  "8": "조여정",
  "9": "조이",
  "10": "강미나",
  "11": "강타",
  "12": "김아중",
  "13": "김범",
  "14": "김구라",
  "15": "김종국",
  "16": "김남길",
  "17": "김서형",
  "18": "김소혜",
  "19": "김수로",
  "20": "최자",
  "21": "김성규",
  "22": "김성령",
  "23": "김태리",
  "24": "김연자",
  "25": "김영광",
  "26": "고두심",
  "27": "고현정",
  "28": "고준희",
  "29": "권혁수",
  "30": "권율",
  "31": "최진혁",
  "32": "규현",
  "33": "이기광",
  "34": "닉쿤",
  "35": "옹성우",
  "36": "설현",
  "37": "태연",
  "38": "TOP",
  "39": "뷔",
  "40": "유승호",
  "41": "최시원",
  "42": "휘인",
  "43": "휘성",
  "44": "장윤정",
  "45": "전혜빈",
  "46": "트와이스 지효",
  "47": "지상렬",
  "48": "지석진",
  "49": "지숙",
  "50": "주이",
  "51": "준호",
  "52": "정은지",
  "53": "정국",
  "54": "정해인",
  "55": "정형돈",
  "56": "비",
  "57": "정진운",
  "58": "정경호",
  "59": "정려원",
  "60": "정우성",
  "61": "정유미",
  "62": "김해숙",
  "63": "김희애",
  "64": "김희철",
  "65": "김희선",
  "66": "김향기",
  "67": "김혜수",
  "68": "김지원",
  "69": "김유정",
  "70": "K.Will",
  "71": "나나",
  "72": "남주혁",
  "73": "남궁민",
  "74": "낸시",
  "75": "옥택연",
  "76": "팔로알토",
  "77": "선미",
  "78": "타블로",
  "79": "태민",
  "80": "태양",
  "81": "테이",
  "82": "타이거JK",
  "83": "육중완",
  "84": "금보라",
  "85": "기태영",
  "86": "공형진",
  "87": "공승연",
  "88": "강동원",
  "89": "강하늘",
  "90": "강호동",
  "91": "강기영",
  "92": "강민경",
  "93": "강소라",
  "94": "김동희",
  "95": "김동완",
  "96": "김고은",
  "97": "김래원",
  "98": "김사랑",
  "99": "김수현",
  "100": "김숙",
  "101": "김태희",
  "102": "김완선",
  "103": "김연아",
  "104": "고창석",
  "105": "고경표",
  "106": "고세원",
  "107": "이달의소녀 고원",
  "108": "아이즈원 권은비",
  "109": "경리",
  "110": "백현",
  "111": "박서준",
  "112": "보아",
  "113": "차은우",
  "114": "가인",
  "115": "개리",
  "116": "금새록",
  "117": "기리보이",
  "118": "공찬",
  "119": "공유",
  "120": "구혜선",
  "121": "하니",
  "122": "한예슬",
  "123": "아이유",
  "124": "전지현",
  "125": "강부자",
  "126": "강다니엘",
  "127": "강한나",
  "128": "강혜정",
};
const IMAGE_SIZE = 32;
TOPK_PREDICTIONS = 20;
const predictionsElement = document.getElementById("predictions");
let loader__message = document.querySelector("#loader__message");

async function predict() {
  loader__message.innerHTML = "데이터베이스에서 닮은 얼굴을 찾습니다";
  const model = await tf.loadLayersModel("js/model/model.json");

  const imgElement = document.querySelector("#canvasOutput");
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(255);
    // Normalize the image from [0, 255] to [0, 1].
    const normalized = img.div(offset);
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    const pred = model.predict(batched);

    return pred;
  });
  console.log("logits", await logits.data());
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  // Show the classes in the DOM.
  showResults(classes);
  showReturn();
}
function showReturn() {
  const start__btn = document.querySelector(".start__btn");
  start__btn.style.display = "block";
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

  const predictionContainer = document.createElement("ul");
  predictionContainer.className = "skills-bar-container";

  for (let i = 0; i < 5; i++) {
    const celeb = classes[i].className;
    const prob = (classes[i].probability * 100).toFixed(2) + "%";

    const row = document.createElement("li");
    row.className = "row";

    const classElement = document.createElement("div");
    classElement.className = "progressbar-title";

    const classTitle = document.createElement("h3");
    classTitle.innerText = celeb;

    classElement.appendChild(classTitle);

    const probsElement = document.createElement("span");
    probsElement.id = celeb + "-percent";
    probsElement.className = "percent";

    classElement.appendChild(probsElement);
    row.appendChild(classElement);
    const barContainer = document.createElement("div");
    barContainer.className = "bar-container";
    const progressbar = document.createElement("progress");
    progressbar.id = "progress-" + celeb;
    progressbar.className = "progressbar " + "progressbar-" + i;
    progressbar.value = 0;
    progressbar.max = "100";

    barContainer.appendChild(progressbar);
    row.appendChild(barContainer);

    predictionContainer.appendChild(row);

    probs[celeb] = prob;
  }
  var multiply = 4;
  for (let [celeb, percent] of Object.entries(probs)) {
    var delay = 700;

    setTimeout(function () {
      document.getElementById(celeb + "-percent").innerHTML = percent;
      document.getElementById("progress-" + celeb).value = percent.replace(
        "%",
        ""
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
  loader__message.innerHTML = "사진을 찾았습니다! 얼굴을 인식합니다.";
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      loading();
      $(".image-upload-wrap").hide();
      $(".file-upload-content").show();
      $(".image-title").html(input.files[0].name);
      let utils = new Utils("errorMessage");
      utils.loadImageToCanvas(e.target.result, "canvasInput");
      utils.loadOpenCv(() => {
        let faceCascadeFile = "haarcascade_frontalface_alt.xml";
        utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
          loader__message.innerHTML = "얼굴 인식중.....";
          faceDetect();
        });
      });
    };

    reader.readAsDataURL(input.files[0]);
  } else {
    alert("img");
  }
}
function faceDetect() {
  let src = cv.imread("canvasInput");
  let gray = new cv.Mat();
  let resultImg = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
  let faces = new cv.RectVector();
  let faceCascade = new cv.CascadeClassifier();
  let msize = new cv.Size(0, 0);
  let dsize = new cv.Size(32, 32);
  // load pre-trained classifiers
  faceCascade.load("haarcascade_frontalface_alt.xml");
  //Call the time-consuming function

  faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

  // detect faces
  // You can try more different parameters
  try {
    src = gray.roi(faces.get(0));
  } catch (error) {
    alert("얼굴인식 실패;;;;");
    location.reload();
  }
  cv.resize(src, src, dsize, 0, 0, cv.INTER_AREA);
  // cv.crop(src, point1, point2, [255, 0, 0, 255]);
  // }
  cv.imshow("canvasOutput", src);

  predict().then(load_finish);
  src.delete();
  gray.delete();
  faceCascade.delete();
  faces.delete();
}
function loading() {
  const loader = document.querySelector("#loader-wrapper");
  loader.style.display = "flex";
}
function load_finish() {
  const loader = document.querySelector("#loader-wrapper");
  loader.style.display = "none";
}
