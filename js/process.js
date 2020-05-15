// import * as tf from '@tensorflow/tfjs';

async function init() {
  const model = await tf.loadLayersModel('../celeb/tfjs_artifacts/model.json');
  const img = document.getElementById('_img');
  const tfImg = tf.browser.fromPixels(img);
  const smalImg = tf.image.resizeBilinear(tfImg, [64, 64]);
  const resized = tf.cast(smalImg, 'float32');
  const t4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 64, 64, 3]);

  const labelContainer = document.getElementById('label-container');
  for (let i = 0; i < 10; i++) {
    // and class labels
    labelContainer.appendChild(document.createElement('div'));
  }
  //const example = tf.fromPixels(img); // for example
  const prediction = await model.predict(t4d, true);
  console.log(model.summary());
  const divv = document.getElementById('_div');
  console.log();
  divv.innerHTML = prediction.dataSync();
}
init();
