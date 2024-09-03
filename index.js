import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { createCanvas, loadImage } from 'canvas';

(async () => {
  // Load the model
  const model = await cocoSsd.load();

  // Load the image using canvas
  const img = await loadImage('./assets/cactus.jpeg');
  
  // Create a canvas and draw the image on it
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, img.width, img.height);

  // Convert canvas to Tensor
  const tensor = tf.browser.fromPixels(canvas);

  // Classify the image
  const predictions = await model.detect(tensor);
  
  console.log('Predictions:');
  console.log(predictions);
})();
