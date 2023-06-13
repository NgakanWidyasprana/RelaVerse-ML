const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const sastrawi = require('sastrawijs');
const axios = require('axios');
const path = require('path');

let model;
let tokenizer;
const stemmer = new sastrawi.Stemmer();

let stopwordSet = new Set();

const updateStopwordSet = async () => {
  try {
    const response = await axios.get('https://raw.githubusercontent.com/NgakanWidyasprana/title-campaign-classification/main/Dataset/stopwordbahasa.csv');
    const rows = response.data.split('\n');
    rows.forEach((word) => stopwordSet.add(word));
    console.log('Stopwords updated successfully');
  } catch (error) {
    console.error('Error updating stopwords:', error.message);
  }
};

const loadModel = async () => {
  try {
    const modelPath = path.resolve(__dirname, '../models/model.json');
    const tokenizerPath = path.resolve(__dirname, '../models/tokenizer.json');
    model = await tf.loadLayersModel('file://' + modelPath);
    tokenizer = JSON.parse(await fs.readFile(tokenizerPath, 'utf8'));
    console.log('Model and tokenizer loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error.message);
  }
};

const cleanText = (text) => {
  const cleanedText = text.toLowerCase().replace(/[^a-zA-Z]/g, ' ');
  const withoutStopwords = removeStopwords(cleanedText);
  const stemmedText = stemText(withoutStopwords);
  return stemmedText;
};

const removeStopwords = (text) => {
  const words = text.split(' ');
  const filteredWords = words.filter((word) => !stopwordSet.has(word));
  return filteredWords.join(' ');
};

const stemText = (text) => {
  return stemmer.stem(text);
};

const tokenizeAndPad = (text, maxLength = 20) => {
  const words = text.split(' ');
  const tokens = words.map((word) => tokenizer[word] || tokenizer["<XXX>"]);
  const paddedTokens = Array(maxLength).fill(0);
  tokens.forEach((token, index) => {
    if (index < maxLength) {
      paddedTokens[index] = token;
    }
  });
  return paddedTokens;
};

const makePrediction = async (text) => {
  if (!model || !tokenizer) {
    console.error('Model or tokenizer has not been loaded');
    return;
  }

  const cleanedText = cleanText(text);
  console.log('Cleaned text:', cleanedText);
  const tokens = tokenizeAndPad(cleanedText);
  console.log('Tokens:', tokens);
  const inputTensor = tf.tensor([tokens]);
  const prediction = model.predict(inputTensor);
  console.log('Prediction:', prediction);
  const predictionData = await prediction.data();
  console.log('Prediction data:', predictionData);
  return predictionData[0];
};

const initialize = async () => {
  try {
    await updateStopwordSet();
    await loadModel();
  } catch (error) {
    console.error('Initialization error:', error.message);
  }
};

initialize();

module.exports = { makePrediction };