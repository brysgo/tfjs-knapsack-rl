/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from "./tensorflow";
import * as tfvis from "@tensorflow/tfjs-vis";

import { Knapsack } from "./knapsack";
import { SaveablePolicyNetwork } from "./index";
import { mean, sum } from "./utils";

const appStatus = document.getElementById("app-status");
const storedModelStatusInput = document.getElementById("stored-model-status");
const hiddenLayerSizesInput = document.getElementById("hidden-layer-sizes");
const createModelButton = document.getElementById("create-model");
const deleteStoredModelButton = document.getElementById("delete-stored-model");
const knapsackCanvas = document.getElementById("knapsack-canvas");

const numIterationsInput = document.getElementById("num-iterations");
const gamesPerIterationInput = document.getElementById("games-per-iteration");
const discountRateInput = document.getElementById("discount-rate");
const maxStepsPerGameInput = document.getElementById("max-steps-per-game");
const learningRateInput = document.getElementById("learning-rate");
const renderDuringTrainingCheckbox = document.getElementById(
  "render-during-training"
);

const trainButton = document.getElementById("train");
const testButton = document.getElementById("test");
const iterationStatus = document.getElementById("iteration-status");
const iterationProgress = document.getElementById("iteration-progress");
const trainStatus = document.getElementById("train-status");
const trainSpeed = document.getElementById("train-speed");
const trainProgress = document.getElementById("train-progress");

const stepsContainer = document.getElementById("steps-container");

// Module-global instance of policy network.
let policyNet;
let stopRequested = false;

/**
 * Display a message to the info div.
 *
 * @param {string} message The message to be displayed.
 */
function logStatus(message) {
  appStatus.textContent = message;
}

// Objects and functions to support display of cart pole status during training.
let renderDuringTraining = true;
export async function maybeRenderDuringTraining(knapsack) {
  if (renderDuringTraining) {
    renderKnapsack(knapsack, knapsackCanvas);
    await tf.nextFrame(); // Unblock UI thread.
  }
}

/**
 * A function invoked at the end of every game during training.
 *
 * @param {number} gameCount A count of how many games has completed so far in
 *   the current iteration of training.
 * @param {number} totalGames Total number of games to complete in the current
 *   iteration of training.
 */
export function onGameEnd(gameCount, totalGames) {
  iterationStatus.textContent = `Game ${gameCount} of ${totalGames}`;
  iterationProgress.value = (gameCount / totalGames) * 100;
  if (gameCount === totalGames) {
    iterationStatus.textContent = "Updating weights...";
  }
}

/**
 * A function invokved at the end of a training iteration.
 *
 * @param {number} iterationCount A count of how many iterations has completed
 *   so far in the current round of training.
 * @param {*} totalIterations Total number of iterations to complete in the
 *   current round of training.
 */
function onIterationEnd(iterationCount, totalIterations) {
  trainStatus.textContent = `Iteration ${iterationCount} of ${totalIterations}`;
  trainProgress.value = (iterationCount / totalIterations) * 100;
}

// Objects and function to support the plotting of game steps during training.
let meanStepValues = [];
function plotSteps() {
  tfvis.render.linechart(
    stepsContainer,
    { values: meanStepValues },
    {
      xLabel: "Training Iteration",
      yLabel: "Mean Steps Per Game",
      width: 400,
      height: 300,
    }
  );
}

function disableModelControls() {
  trainButton.textContent = "Stop";
  testButton.disabled = true;
  deleteStoredModelButton.disabled = true;
}

function enableModelControls() {
  trainButton.textContent = "Train";
  testButton.disabled = false;
  deleteStoredModelButton.disabled = false;
}

function drawBar(
  ctx,
  upperLeftCornerX,
  upperLeftCornerY,
  width,
  height,
  color
) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.fillRect(upperLeftCornerX, upperLeftCornerY, width, height);
  ctx.restore();
}

function heatMapColor(value) {
  var h = (1.0 - value) * 240;
  return "hsl(" + h + ", 100%, 50%)";
}

const SEPARATOR = "separator";
function drawStackedBar(
  ctx,
  upperLeftCornerX,
  upperLeftCornerY,
  width,
  height,
  values,
  total,
  maxValue
) {
  const separatorValue = (10 * total) / width;
  const totalWithSeparator = total + separatorValue;
  let runningTotal = 0;
  values.forEach((value, i) => {
    if (value === SEPARATOR) {
      drawBar(
        ctx,
        width * (runningTotal / totalWithSeparator) + upperLeftCornerX,
        upperLeftCornerY,
        width * (separatorValue / totalWithSeparator),
        height,
        "black"
      );
      runningTotal += separatorValue;
    } else {
      drawBar(
        ctx,
        width * (runningTotal / totalWithSeparator) + upperLeftCornerX,
        upperLeftCornerY,
        width * (value / totalWithSeparator),
        height,
        heatMapColor(value / maxValue)
      );
      runningTotal += value;
    }
  });
}

/**
 * Render the current state of the system on an HTML canvas.
 *
 * @param {Knapsack} knapsack The instance of knapsack system to render.
 * @param {HTMLCanvasElement} canvas The instance of HTMLCanvasElement on which
 *   the rendering will happen.
 */
function renderKnapsack(knapsack, canvas) {
  if (!knapsack) return;
  if (!canvas.style.display) {
    canvas.style.display = "block";
  }
  // knapsack.lastStateTensor;
  // knapsack.cursor;

  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  const halfW = canvas.width / 2;

  context.font = "24px serif";
  let totalCostIn = 0,
    totalValueIn = 0,
    totalCostOut = 0,
    totalValueOut = 0,
    totalROI = 0,
    maxCost = 0,
    maxValue = 0,
    maxROI = 0;
  const costs = [SEPARATOR];
  const values = [SEPARATOR];
  const rois = [SEPARATOR];
  knapsack.items.arraySync().forEach(([cost, value, inKnapsack], i) => {
    const ROI = value / cost;
    const method = inKnapsack ? "unshift" : "push";
    costs[method](cost);
    values[method](value);
    rois[method](ROI);
    if (inKnapsack) {
      totalCostIn += cost;
      totalValueIn += value;
    } else {
      totalCostOut += cost;
      totalValueOut += value;
    }
    totalROI += ROI;
    maxCost = Math.max(maxCost, cost);
    maxValue = Math.max(maxValue, value);
    maxROI = Math.max(maxROI, ROI);
  });

  context.fillText("in", 100, 25);
  context.fillText("out", 450, 25);
  context.fillText("value", 10, 100);
  context.fillText("cost", 10, 200);
  context.fillText("ROI", 10, 300);
  drawStackedBar(
    context,
    100,
    50,
    400,
    100,
    values,
    totalValueIn + totalValueOut,
    maxValue
  );
  drawStackedBar(
    context,
    100,
    150,
    400,
    100,
    costs,
    totalCostIn + totalCostOut,
    maxCost
  );
  drawStackedBar(context, 100, 250, 400, 100, rois, totalROI, maxROI);

  context.fillText("score", 10, 400);
  context.fillText(knapsack.value(), 100, 400);
}

async function updateUIControlState() {
  const modelInfo = await SaveablePolicyNetwork.checkStoredModelStatus();
  if (modelInfo == null) {
    storedModelStatusInput.value = "No stored model.";
    deleteStoredModelButton.disabled = true;
  } else {
    storedModelStatusInput.value = `Saved@${modelInfo.dateSaved.toISOString()}`;
    deleteStoredModelButton.disabled = false;
    createModelButton.disabled = true;
  }
  createModelButton.disabled = policyNet != null;
  hiddenLayerSizesInput.disabled = policyNet != null;
  trainButton.disabled = policyNet == null;
  testButton.disabled = policyNet == null;
  renderDuringTrainingCheckbox.checked = renderDuringTraining;
}

export async function setUpUI() {
  const knapsack = new Knapsack(true);

  if ((await SaveablePolicyNetwork.checkStoredModelStatus()) != null) {
    policyNet = await SaveablePolicyNetwork.loadModel();
    logStatus("Loaded policy network from IndexedDB.");
    hiddenLayerSizesInput.value = policyNet.hiddenLayerSizes();
  }
  await updateUIControlState();

  renderDuringTrainingCheckbox.addEventListener("change", () => {
    renderDuringTraining = renderDuringTrainingCheckbox.checked;
  });

  createModelButton.addEventListener("click", async () => {
    try {
      const hiddenLayerSizes = hiddenLayerSizesInput.value
        .trim()
        .split(",")
        .map((v) => {
          const num = Number.parseInt(v.trim());
          if (!(num > 0)) {
            throw new Error(
              `Invalid hidden layer sizes string: ` +
                `${hiddenLayerSizesInput.value}`
            );
          }
          return num;
        });
      policyNet = new SaveablePolicyNetwork(hiddenLayerSizes);
      console.log("DONE constructing new instance of SaveablePolicyNetwork");
      await updateUIControlState();
    } catch (err) {
      logStatus(`ERROR: ${err.message}`);
    }
  });

  deleteStoredModelButton.addEventListener("click", async () => {
    if (confirm(`Are you sure you want to delete the locally-stored model?`)) {
      await policyNet.removeModel();
      policyNet = null;
      await updateUIControlState();
    }
  });

  trainButton.addEventListener("click", async () => {
    if (trainButton.textContent === "Stop") {
      stopRequested = true;
    } else {
      disableModelControls();

      try {
        const trainIterations = Number.parseInt(numIterationsInput.value);
        if (!(trainIterations > 0)) {
          throw new Error(`Invalid number of iterations: ${trainIterations}`);
        }
        const gamesPerIteration = Number.parseInt(gamesPerIterationInput.value);
        if (!(gamesPerIteration > 0)) {
          throw new Error(
            `Invalid # of games per iterations: ${gamesPerIteration}`
          );
        }
        const maxStepsPerGame = Number.parseInt(maxStepsPerGameInput.value);
        if (!(maxStepsPerGame > 1)) {
          throw new Error(`Invalid max. steps per game: ${maxStepsPerGame}`);
        }
        const discountRate = Number.parseFloat(discountRateInput.value);
        if (!(discountRate > 0 && discountRate < 1)) {
          throw new Error(`Invalid discount rate: ${discountRate}`);
        }
        const learningRate = Number.parseFloat(learningRateInput.value);

        logStatus(
          "Training policy network... Please wait. " +
            "Network is saved to IndexedDB at the end of each iteration."
        );
        const optimizer = tf.train.adam(learningRate);

        meanStepValues = [];
        onIterationEnd(0, trainIterations);
        let t0 = new Date().getTime();
        stopRequested = false;
        for (let i = 0; i < trainIterations; ++i) {
          const gameSteps = await policyNet.train(
            knapsack,
            optimizer,
            discountRate,
            gamesPerIteration,
            maxStepsPerGame
          );
          const t1 = new Date().getTime();
          const stepsPerSecond = sum(gameSteps) / ((t1 - t0) / 1e3);
          t0 = t1;
          trainSpeed.textContent = `${stepsPerSecond.toFixed(1)} steps/s`;
          meanStepValues.push({ x: i + 1, y: mean(gameSteps) });
          console.log(`# of tensors: ${tf.memory().numTensors}`);
          plotSteps();
          onIterationEnd(i + 1, trainIterations);
          await tf.nextFrame(); // Unblock UI thread.
          await policyNet.saveModel();
          await updateUIControlState();

          if (stopRequested) {
            logStatus("Training stopped by user.");
            break;
          }
        }
        if (!stopRequested) {
          logStatus("Training completed.");
        }
      } catch (err) {
        logStatus(`ERROR: ${err.message}`);
      }
      enableModelControls();
    }
  });

  testButton.addEventListener("click", async () => {
    disableModelControls();
    let isDone = false;
    const knapsack = new Knapsack(true);
    let steps = 0;
    stopRequested = false;
    while (!isDone) {
      steps++;
      tf.tidy(() => {
        const action = policyNet.getActions(knapsack.getStateHistoryTensor());
        logStatus(
          `Test in progress. ` +
            `Action: ${action === 1 ? "<--" : " -->"} (Step ${steps})`
        );
        isDone = knapsack.update(action);
        renderKnapsack(knapsack, knapsackCanvas);
      });
      await tf.nextFrame(); // Unblock UI thread.
      if (stopRequested) {
        break;
      }
    }
    if (stopRequested) {
      logStatus(`Test stopped by user after ${steps} step(s).`);
    } else {
      logStatus(`Test finished. Survived ${steps} step(s).`);
    }
    console.log(`# of tensors: ${tf.memory().numTensors}`);
    enableModelControls();
  });
}
