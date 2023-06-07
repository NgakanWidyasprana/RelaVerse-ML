require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');

function loadModel() {
    const MODEL_URL = "/model/text-classification/model.json";
    const model = loadLayersModel(MODEL_URL);
    
    // const embeddings = await model.embed([inputText]);
    // const predictions = await model.predict(embeddings);
    // const positiveSentiment = predictions.arraySync()[0][0];
    
    // const sentimentResult = document.getElementById('classificationResult');
    // sentimentResult.textContent = `Sentiment: ${positiveSentiment}`;
    return model;
}

model = loadModel();
console.log(model.summary());