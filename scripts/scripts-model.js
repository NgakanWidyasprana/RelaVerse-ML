async function textClassifications() {
    const inputText = "Kegiatan Pembersihan Lingkungan";
    
    const MODEL_URL = 'model/text-classification/model.json';
    const model = await use.loadLayersModel(MODEL_URL);
    
    console.log(model.summary());

    const embeddings = await model.embed([inputText]);
    const predictions = await model.predict(embeddings);
    const positiveSentiment = predictions.arraySync()[0][0];
    
    const sentimentResult = document.getElementById('classificationResult');
    sentimentResult.textContent = `Sentiment: ${positiveSentiment}`;
}