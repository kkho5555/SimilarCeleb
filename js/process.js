const IMAGE_DATA = {
  "0": "00_A Yibi",
  "1": "01_A Yirin",
  "2": "02_An Jaehyun",
  "3": "03_An Youngmi",
  "4": "04_Bi Wayi",
  "5": "05_Chan sung",
  "6": "06_chi tah",
  "7": "07_Cho Seho",
  "8": "08_Cho Yeojung",
  "9": "09_Cho Yi",
  "10": "100_Kang Mina",
  "11": "101_Kang Ta",
  "12": "102_Kim Ajoong",
  "13": "103_Kim Bum",
  "14": "104_Kim gura",
  "15": "105_Kim Jongguk",
  "16": "106_Kim Namgil",
  "17": "107_Kim Seohyung",
  "18": "108_Kim Sohye",
  "19": "109_Kim Sooro",
  "20": "10_Choi Ja",
  "21": "110_Kim Sungkyu",
  "22": "111_Kim Sungryung",
  "23": "112_Kim Taeri",
  "24": "113_Kim Yeonja",
  "25": "114_Kim Youngkwang",
  "26": "115_Ko Doosim",
  "27": "116_Ko Hyunjung",
  "28": "117_Ko Junhee",
  "29": "118_Kwon Hyuksoo",
  "30": "119_Kwon Yul",
  "31": "11_Choi Jinhyuk",
  "32": "120_Kyu Hyun",
  "33": "121_Lee Kikwang",
  "34": "122_Nickun",
  "35": "123_Ong sungu",
  "36": "124_Sul hyun",
  "37": "125_Tae youen",
  "38": "126_Top",
  "39": "127_Vui",
  "40": "128_Yu sengho",
  "41": "12_Choi Siwon",
  "42": "13_Hwi In",
  "43": "14_Hwi Sung",
  "44": "15_Jang Yoonjung",
  "45": "16_Jeon Hyebin",
  "46": "17_Ji Hyo",
  "47": "18_Ji Sangryul",
  "48": "19_Ji Seokjin",
  "49": "20_Ji Sook",
  "50": "21_Joo Yi",
  "51": "22_Jun Ho",
  "52": "23_Jung Eunji",
  "53": "24_Jung Guk",
  "54": "25_Jung Haein",
  "55": "26_Jung Hyungdon",
  "56": "27_Jung Jihoon",
  "57": "28_Jung Jinwoon",
  "58": "29_Jung Kyungho",
  "59": "30_Jung Ryeowon",
  "60": "31_Jung Woosung",
  "61": "32_Jung Yumi",
  "62": "33_Kim Haesook",
  "63": "34_Kim Heeae",
  "64": "35_Kim Heechul",
  "65": "36_Kim Heesun",
  "66": "37_Kim Hyangki",
  "67": "38_Kim Hyesoo",
  "68": "39_Kim Jiwon",
  "69": "40_Kim Yujung",
  "70": "41_Kwill",
  "71": "42_Na Na",
  "72": "43_Nam Juhyuk",
  "73": "44_Namgung Min",
  "74": "45_Nancy",
  "75": "46_Ok Taek Yeon",
  "76": "47_Paloalto",
  "77": "48_Sun Mi",
  "78": "49_tablo",
  "79": "50_Tae Min",
  "80": "51_Tae Yang",
  "81": "52_tei",
  "82": "53_tiger jk",
  "83": "54_Yuk Joongwan",
  "84": "55_Geum Bora",
  "85": "56_Gi Taeyoung",
  "86": "57_Gong Hyungjin",
  "87": "58_Gong Seungyeon",
  "88": "59_Kang Dongwon",
  "89": "60_Kang Haneul",
  "90": "61_Kang Hodong",
  "91": "62_Kang Kiyoung",
  "92": "63_Kang Minkyung",
  "93": "64_Kang Sora",
  "94": "65_Kim Donghee",
  "95": "66_Kim Dongwan",
  "96": "67_Kim Goeun",
  "97": "68_Kim Raewon",
  "98": "69_Kim Sarang",
  "99": "70_Kim Soohyun",
  "100": "71_Kim Sook",
  "101": "72_Kim Taehee",
  "102": "73_Kim Wansun",
  "103": "74_Kim Yeona",
  "104": "75_Ko Changseok",
  "105": "76_Ko Kyungpyo",
  "106": "77_Ko Sewon",
  "107": "78_Ko Won",
  "108": "79_Kwon Eunbi",
  "109": "80_Kyung Ri",
  "110": "81_Baek hyun",
  "111": "82_Bak seojun",
  "112": "83_Boa",
  "113": "84_Cha eunwoo",
  "114": "85_Ga in",
  "115": "86_Gae Ri",
  "116": "87_Geum Saerok",
  "117": "88_Gi Riboy",
  "118": "89_Gong chan",
  "119": "90_Gong Yu",
  "120": "91_Gu Hyesun",
  "121": "92_Ha nee",
  "122": "93_Han yesel",
  "123": "94_IU",
  "124": "95_Jun jihyun",
  "125": "96_Kang Booja",
  "126": "97_Kang daniel",
  "127": "98_Kang Hanna",
  "128": "99_Kang Hyejung",
};
const IMAGE_SIZE = 32;
TOPK_PREDICTIONS = 20;
const predictionsElement = document.getElementById("predictions");
let loader__message = document.querySelector("#loader__message");
async function predict() {
  loader__message.innerHTML = "데이터베이스에서 닮은 얼굴을 찾습니다";
  const model = await tf.loadLayersModel("js/model2/model.json");

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
