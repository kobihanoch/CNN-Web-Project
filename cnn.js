const LABELS = [
  { id: "circle", name: "עיגול" },
  { id: "square", name: "ריבוע" },
  { id: "triangle", name: "משולש" }
];

const STORAGE_KEY = "cnnShapesModelV3";
const SAMPLES_KEY = "cnnShapesSamplesV2";
const PRETRAINED_KEY = "cnnShapesPretrainedV3";
const INPUT_SIZE = 16;
const CLASS_COUNT = LABELS.length;

let model;
let samples = [];
let drawing = false;

function el(id) {
  return document.getElementById(id);
}

function dot(weights, values, bias) {
  let sum = bias;
  for (let i = 0; i < weights.length; i++) {
    sum += weights[i] * values[i];
  }
  return sum;
}

function copyRows(rows) {
  const copy = [];
  for (let i = 0; i < rows.length; i++) {
    copy.push(rows[i].slice());
  }
  return copy;
}

function bestIndex(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

const canvas = el("drawCanvas");
const ctx = canvas.getContext("2d");
const statusText = el("modelStatus");
const sampleCount = el("sampleCount");
const predictionText = el("predictionText");
const lossValue = el("lossValue");
const accuracyValue = el("accuracyValue");
const trainRunsValue = el("trainRuns");
const activeConfigText = el("activeConfig");

const inputs = {
  layers: el("layerCount"),
  filters: el("filterCount"),
  filterSize: el("filterSize"),
  neurons: el("hiddenNeurons"),
  lr: el("learningRate"),
  epochs: el("epochs"),
  label: el("labelSelect")
};

const bars = {
  circle: [el("probCircle"), el("probCircleText")],
  square: [el("probSquare"), el("probSquareText")],
  triangle: [el("probTriangle"), el("probTriangleText")]
};

function readConfig() {
  return {
    layers: Number(inputs.layers.value),
    filters: Number(inputs.filters.value),
    filterSize: Number(inputs.filterSize.value),
    neurons: Number(inputs.neurons.value),
    learningRate: Number(inputs.lr.value),
    trainRuns: 0
  };
}

function featureCount(config) {
  return config.layers * config.filters;
}

function randomWeight(scale) {
  return (Math.random() - 0.5) * scale;
}

function createKernel(size, layer, filter) {
  const mid = Math.floor(size / 2);
  const kernel = [];

  for (let y = 0; y < size; y++) {
    const row = [];
    for (let x = 0; x < size; x++) {
      let value = randomWeight(0.16);

      if (filter % 5 === 0 && x === mid) value += 0.25;
      if (filter % 5 === 1 && y === mid) value += 0.25;
      if (filter % 5 === 2 && x === y) value += 0.25;
      if (filter % 5 === 3 && x + y === size - 1) value += 0.25;
      if (filter % 5 === 4 && (x === 0 || y === 0 || x === size - 1 || y === size - 1)) value += 0.12;

      row.push(value / (layer + 1));
    }
    kernel.push(row);
  }

  return kernel;
}

function createModel(config) {
  const convFeatures = featureCount(config);
  const convKernels = [];
  const convBiases = [];
  const hiddenWeights = [];
  const weights = [];

  for (let layer = 0; layer < config.layers; layer++) {
    const layerKernels = [];
    for (let filter = 0; filter < config.filters; filter++) {
      layerKernels.push(createKernel(config.filterSize, layer, filter));
    }
    convKernels.push(layerKernels);
    convBiases.push(Array(config.filters).fill(0));
  }

  for (let neuron = 0; neuron < config.neurons; neuron++) {
    const row = [];
    for (let i = 0; i < convFeatures; i++) row.push(randomWeight(0.24));
    hiddenWeights.push(row);
  }

  for (let label = 0; label < CLASS_COUNT; label++) {
    const row = [];
    for (let i = 0; i < config.neurons; i++) row.push(randomWeight(0.24));
    weights.push(row);
  }

  return {
    version: 3,
    config,
    convKernels,
    convBiases,
    hiddenWeights,
    hiddenBiases: Array(config.neurons).fill(0),
    weights,
    biases: [0, 0, 0]
  };
}

function saveModel() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(model));
}

function saveSamples() {
  localStorage.setItem(SAMPLES_KEY, JSON.stringify(samples));
  sampleCount.textContent = `דוגמאות משתמש: ${samples.length}`;
}

function syncInputs() {
  inputs.layers.value = model.config.layers;
  inputs.filters.value = model.config.filters;
  inputs.filterSize.value = model.config.filterSize;
  inputs.neurons.value = model.config.neurons;
  inputs.lr.value = model.config.learningRate;
  trainRunsValue.textContent = model.config.trainRuns || 0;
  activeConfigText.textContent =
    `פרמטרים מקובעים במודל: ${model.config.layers} שכבות, ` +
    `${model.config.filters} פילטרים בכל שכבה, פילטר ${model.config.filterSize}x${model.config.filterSize}, ` +
    `${model.config.neurons} נוירונים, קצב למידה ${model.config.learningRate}. ` +
    `כדי לשנות אותם לוחצים שוב על "קבע פרמטרים" ונבנה מודל חדש.`;
}

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const point = event.touches ? event.touches[0] : event;
  return {
    x: (point.clientX - rect.left) * canvas.width / rect.width,
    y: (point.clientY - rect.top) * canvas.height / rect.height
  };
}

function startDraw(event) {
  event.preventDefault();
  drawing = true;
  const point = pointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(point.x, point.y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const point = pointerPosition(event);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
}

function stopDraw() {
  drawing = false;
}

function canvasToInput() {
  const small = document.createElement("canvas");
  small.width = INPUT_SIZE;
  small.height = INPUT_SIZE;
  const smallCtx = small.getContext("2d");
  smallCtx.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const data = smallCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const image = [];

  for (let y = 0; y < INPUT_SIZE; y++) {
    const row = [];
    for (let x = 0; x < INPUT_SIZE; x++) {
      const i = (y * INPUT_SIZE + x) * 4;
      row.push(1 - (data[i] + data[i + 1] + data[i + 2]) / 765);
    }
    image.push(row);
  }

  return image;
}

function runConvolution(image, kernel, bias) {
  const size = kernel.length;
  const pad = Math.floor(size / 2);
  const values = [];
  const active = [];
  let total = 0;

  for (let y = 0; y < INPUT_SIZE; y++) {
    const valueRow = [];
    const activeRow = [];
    for (let x = 0; x < INPUT_SIZE; x++) {
      let sum = bias;
      for (let ky = 0; ky < size; ky++) {
        for (let kx = 0; kx < size; kx++) {
          const iy = y + ky - pad;
          const ix = x + kx - pad;
          if (iy >= 0 && iy < INPUT_SIZE && ix >= 0 && ix < INPUT_SIZE) {
            sum += image[iy][ix] * kernel[ky][kx];
          }
        }
      }
      const value = Math.max(0, sum);
      valueRow.push(value);
      activeRow.push(sum > 0);
      total += value;
    }
    values.push(valueRow);
    active.push(activeRow);
  }

  return {
    map: values,
    active,
    pooled: total / (INPUT_SIZE * INPUT_SIZE)
  };
}

function forwardPass(image) {
  const convFeatures = [];
  const convCache = [];

  for (let layer = 0; layer < model.config.layers; layer++) {
    for (let filter = 0; filter < model.config.filters; filter++) {
      const result = runConvolution(
        image,
        model.convKernels[layer][filter],
        model.convBiases[layer][filter]
      );
      convFeatures.push(result.pooled);
      convCache.push({ layer, filter, active: result.active });
    }
  }

  const hiddenRaw = [];
  const hidden = [];
  for (let neuron = 0; neuron < model.config.neurons; neuron++) {
    const raw = dot(model.hiddenWeights[neuron], convFeatures, model.hiddenBiases[neuron]);
    hiddenRaw.push(raw);
    hidden.push(Math.max(0, raw));
  }

  const scores = [];
  for (let label = 0; label < CLASS_COUNT; label++) {
    scores.push(dot(model.weights[label], hidden, model.biases[label]));
  }

  return {
    image,
    convFeatures,
    convCache,
    hiddenRaw,
    hidden,
    probabilities: softmax(scores)
  };
}

function softmax(scores) {
  const max = scores[bestIndex(scores)];
  const exps = [];
  let sum = 0;

  for (let i = 0; i < scores.length; i++) {
    const value = Math.exp(scores[i] - max);
    exps.push(value);
    sum += value;
  }

  for (let i = 0; i < exps.length; i++) {
    exps[i] = exps[i] / sum;
  }

  return exps;
}

function predictImage(image) {
  return forwardPass(image);
}

function trainSample(image, labelIndex) {
  const result = forwardPass(image);
  const target = [0, 0, 0];
  const outputErrors = [];
  const hiddenErrors = [];
  const convFeatureErrors = [];

  target[labelIndex] = 1;

  for (let label = 0; label < CLASS_COUNT; label++) {
    outputErrors.push(result.probabilities[label] - target[label]);
  }

  const oldOutputWeights = copyRows(model.weights);
  const oldHiddenWeights = copyRows(model.hiddenWeights);

  for (let label = 0; label < CLASS_COUNT; label++) {
    for (let i = 0; i < result.hidden.length; i++) {
      model.weights[label][i] -= model.config.learningRate * outputErrors[label] * result.hidden[i];
    }
    model.biases[label] -= model.config.learningRate * outputErrors[label];
  }

  for (let neuron = 0; neuron < result.hidden.length; neuron++) {
    let error = 0;
    for (let label = 0; label < CLASS_COUNT; label++) {
      error += outputErrors[label] * oldOutputWeights[label][neuron];
    }
    hiddenErrors.push(result.hiddenRaw[neuron] > 0 ? error : 0);
  }

  for (let feature = 0; feature < result.convFeatures.length; feature++) {
    let error = 0;
    for (let neuron = 0; neuron < hiddenErrors.length; neuron++) {
      error += hiddenErrors[neuron] * oldHiddenWeights[neuron][feature];
    }
    convFeatureErrors.push(error);
  }

  for (let neuron = 0; neuron < model.config.neurons; neuron++) {
    for (let feature = 0; feature < result.convFeatures.length; feature++) {
      model.hiddenWeights[neuron][feature] -= model.config.learningRate * hiddenErrors[neuron] * result.convFeatures[feature];
    }
    model.hiddenBiases[neuron] -= model.config.learningRate * hiddenErrors[neuron];
  }

  const pixelCount = INPUT_SIZE * INPUT_SIZE;
  for (let featureIndex = 0; featureIndex < result.convCache.length; featureIndex++) {
    const cache = result.convCache[featureIndex];
    const kernel = model.convKernels[cache.layer][cache.filter];
    const size = kernel.length;
    const pad = Math.floor(size / 2);
    const convError = convFeatureErrors[featureIndex] / pixelCount;
    let biasGradient = 0;

    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        if (!cache.active[y][x]) continue;
        biasGradient += convError;
        for (let ky = 0; ky < size; ky++) {
          for (let kx = 0; kx < size; kx++) {
            const iy = y + ky - pad;
            const ix = x + kx - pad;
            if (iy >= 0 && iy < INPUT_SIZE && ix >= 0 && ix < INPUT_SIZE) {
              kernel[ky][kx] -= model.config.learningRate * convError * image[iy][ix];
            }
          }
        }
      }
    }

    model.convBiases[cache.layer][cache.filter] -= model.config.learningRate * biasGradient;
  }

  const guess = bestIndex(result.probabilities);
  return {
    loss: -Math.log(result.probabilities[labelIndex] + 0.000001),
    correct: guess === labelIndex
  };
}

function syntheticShape(labelIndex) {
  const image = [];
  const cx = 7.5 + (Math.random() - 0.5) * 2;
  const cy = 7.5 + (Math.random() - 0.5) * 2;
  const r = 4.5 + Math.random();

  for (let y = 0; y < INPUT_SIZE; y++) {
    const row = [];
    for (let x = 0; x < INPUT_SIZE; x++) {
      const dx = x - cx;
      const dy = y - cy;
      let onShape = false;

      if (labelIndex === 0) {
        onShape = Math.abs(Math.sqrt(dx * dx + dy * dy) - r) < 1;
      } else if (labelIndex === 1) {
        onShape = Math.abs(Math.abs(dx) - r) < 1 && Math.abs(dy) < r ||
          Math.abs(Math.abs(dy) - r) < 1 && Math.abs(dx) < r;
      } else {
        const line1 = Math.abs(dy + r - Math.abs(dx) * 1.7) < 1;
        const line2 = Math.abs(dy - r) < 1 && Math.abs(dx) < r;
        onShape = line1 || line2;
      }

      row.push(onShape ? 1 : Math.random() < 0.01 ? 0.6 : 0);
    }
    image.push(row);
  }

  return { image, labelIndex };
}

function trainingSet() {
  const set = [];

  for (let label = 0; label < CLASS_COUNT; label++) {
    for (let i = 0; i < 22; i++) set.push(syntheticShape(label));
  }

  for (let i = 0; i < samples.length; i++) {
    set.push(samples[i]);
  }

  return set.sort(() => Math.random() - 0.5);
}

async function trainModel(epochs) {
  statusText.textContent = "מאמן...";

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const set = trainingSet();
    let loss = 0;
    let correct = 0;

    for (let i = 0; i < set.length; i++) {
      const sample = set[i];
      const result = trainSample(sample.image, sample.labelIndex);
      loss += result.loss;
      if (result.correct) correct++;
    }

    lossValue.textContent = (loss / set.length).toFixed(3);
    accuracyValue.textContent = `${Math.round(correct / set.length * 100)}%`;
    if (epoch % 10 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  model.config.trainRuns = (model.config.trainRuns || 0) + 1;
  saveModel();
  syncInputs();
  statusText.textContent = "המודל אומן ונשמר ב-LocalStorage";
}

function showPrediction(probabilities) {
  const best = bestIndex(probabilities);

  predictionText.textContent = `המודל מזהה: ${LABELS[best].name} (${Math.round(probabilities[best] * 100)}%)`;
  for (let index = 0; index < CLASS_COUNT; index++) {
    const label = LABELS[index];
    bars[label.id][0].value = probabilities[index];
    bars[label.id][1].textContent = `${Math.round(probabilities[index] * 100)}%`;
  }
}

function addSample() {
  let labelIndex = -1;

  for (let i = 0; i < CLASS_COUNT; i++) {
    if (LABELS[i].id === inputs.label.value) {
      labelIndex = i;
      break;
    }
  }

  samples.push({ image: canvasToInput(), labelIndex });
  saveSamples();
  statusText.textContent = "דוגמה נוספה לאימון";
}

function exportPayload() {
  return {
    version: 3,
    exportedAt: new Date().toISOString(),
    model,
    samples
  };
}

function download(name, text, type) {
  const blob = new Blob([text], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

function exportJson() {
  download("cnn-trained-weights.json", JSON.stringify(exportPayload(), null, 2), "application/json");
}

function exportJs() {
  download("trained-model.js", `window.PRETRAINED_CNN_MODEL = ${JSON.stringify(exportPayload(), null, 2)};\n`, "text/javascript");
}

function importPayload(payload) {
  if (!payload || !payload.model || !payload.model.convKernels || !payload.model.hiddenWeights || !payload.model.weights) {
    return false;
  }
  model = payload.model;
  samples = payload.samples || [];
  saveModel();
  saveSamples();
  localStorage.setItem(PRETRAINED_KEY, payload.exportedAt || "model");
  syncInputs();
  statusText.textContent = "משקלים נטענו ונשמרו ב-LocalStorage";
  return true;
}

function importFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result.trim();
      const start = text.indexOf("{");
      const end = text.lastIndexOf("}");
      const payload = JSON.parse(text.slice(start, end + 1));
      if (!importPayload(payload)) throw new Error();
    } catch {
      statusText.textContent = "קובץ המשקלים לא תקין";
    }
    event.target.value = "";
  };
  reader.readAsText(file);
}

function buildNewModel() {
  model = createModel(readConfig());
  saveModel();
  syncInputs();
  statusText.textContent = "נוצר מודל חדש";
}

function resetAll() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(SAMPLES_KEY);
  localStorage.removeItem(PRETRAINED_KEY);
  samples = [];
  model = createModel(readConfig());
  saveModel();
  saveSamples();
  syncInputs();
  clearCanvas();
  lossValue.textContent = "-";
  accuracyValue.textContent = "-";
  statusText.textContent = "המודל אופס";
}

async function init() {
  clearCanvas();
  samples = JSON.parse(localStorage.getItem(SAMPLES_KEY) || "[]");
  saveSamples();

  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  const embedded = window.PRETRAINED_CNN_MODEL;
  const embeddedMarker = embedded && embedded.exportedAt || "model";

  if (embedded && localStorage.getItem(PRETRAINED_KEY) !== embeddedMarker) {
    importPayload(embedded);
  } else if (saved) {
    model = saved;
    syncInputs();
    statusText.textContent = "מודל נטען מ-LocalStorage";
  } else {
    model = createModel(readConfig());
    syncInputs();
    await trainModel(20);
  }
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDraw);

el("clearCanvasBtn").addEventListener("click", clearCanvas);
el("predictBtn").addEventListener("click", () => showPrediction(predictImage(canvasToInput()).probabilities));
el("addSampleBtn").addEventListener("click", addSample);
el("buildBtn").addEventListener("click", buildNewModel);
el("trainBtn").addEventListener("click", () => trainModel(Number(inputs.epochs.value)));
el("resetBtn").addEventListener("click", resetAll);
el("exportJsonBtn").addEventListener("click", exportJson);
el("exportJsBtn").addEventListener("click", exportJs);
el("importModelInput").addEventListener("change", importFile);

init();
