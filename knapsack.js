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

/**
 * Implementation based on: http://incompleteideas.net/book/code/pole.c
 */

import * as tf from "./tensorflow";
import { pad } from "./utils";

/**
 * Knapsack system simulator.
 *
 * The state is a tensor of shape [2, 2, 3]:
 *
 * [left/right, in/out, cost/value)]
 *
 * The system is controlled through a two actions:
 *
 *   - leftward or rightward curosor jump by 1/2.
 *   - whether to flip in backpack state.
 */
export class Knapsack {
  /**
   * Constructor of Knapsack.
   */
  constructor() {
    // Constants that characterize the system.
    this.itemRange = { min: 50, max: 1000 };
    this.costValueMultiplier = 10;
    this.idleThreshold = 5;
    this.historySize = 20;

    this.setRandomState();
  }

  /**
   * Set the state of the knapsack system randomly.
   */
  setRandomState() {
    // Everything but item length will be normalized
    const numItems = Math.floor(
      this.itemRange.min + Math.random() * this.itemRange.max
    );
    // [numberOfItems, (cost, value,inKnapsack, visitCount)]
    tf.dispose(this.items);
    this.items = tf.tidy(() =>
      tf.concat(
        [
          tf.randomUniform(
            [numItems, 2],
            0,
            this.costValueMultiplier / numItems
          ),
          tf.zeros([numItems, 2]), // keep this zero for better reward
        ],
        1
      )
    );
    tf.dispose(this.normalization);
    this.normalization = {
      visitCounter: 0,
    };
    this.cursor = {
      index: 0, // index of cursor in list
      stride: 0, // track depth of cursor in virtual tree
    };
    this.treeDepth = Math.floor(Math.log(numItems));
    this.idleCount = 0;
  }

  /**
   * Get current state as a tf.Tensor of shape [2, 2, 2].
   * [# batches, left/right, in/out, cost/value)]
   *
   */
  getStateTensor() {
    return tf.tidy(() => {
      const expDistanceFromCursor = tf.exp(
        tf.concat([
          this.cursor.index > 0
            ? tf.linspace(0, 1, this.cursor.index)
            : tf.zeros([0]),
          this.cursor.index < this.items.shape[0]
            ? tf.linspace(1, 0, this.items.shape[0] - this.cursor.index)
            : tf.zeros([0]),
        ])
      );
      const itemOnes = tf.ones([this.items.shape[0]]);
      const [costItems, valueItems] = tf.unstack(this.items, 1);
      const roiItems = valueItems.div(costItems);

      const distanceAdjustedROI = expDistanceFromCursor.mul(roiItems);
      return tf
        .stack(
          [
            pad(this.items.slice(0, this.cursor.index), [
              [0, this.items.shape[0] - this.cursor.index],
              [0, 0],
            ]),
            pad(this.items.slice(this.cursor.index + 1), [
              [this.cursor.index + 1, 0],
              [0, 0],
            ]),
          ].map((itemsPos) => {
            const [
              costPosItems,
              valuePosItems,
              inKnapsackPosItems,
              visitCounter,
            ] = tf.unstack(itemsPos, 1);
            const roiPosItems = valuePosItems.div(costPosItems);
            const distanceAdjustedPosROI = expDistanceFromCursor.mul(
              roiPosItems
            );
            const valueCostPos = tf.stack([
              roiPosItems.div(roiItems.sum()),
              distanceAdjustedPosROI.div(distanceAdjustedROI.sum()),
              itemOnes.div(this.items.shape[0]),
              visitCounter.div(this.normalization.visitCounter),
            ]);
            return tf.stack([
              tf.mul(valueCostPos, inKnapsackPosItems),
              tf.mul(valueCostPos, tf.scalar(1).sub(inKnapsackPosItems)),
            ]);
          })
        )
        .sum(-1);
    });
  }

  /**
   * Update the knapsack system using an action.
   * @param {[ number, number ]} actions
   *   A probLeft > 0 leads to a rightward move of half the item count
   *   A probLeft <= 0 leads to a leftward move of half the item count
   *   A flipProb > 0 leads to flipping the item's in backpack state
   *   A flipProb < 0 leads to leaving the item's state as is
   */
  update([probLeft, probIn]) {
    const leftOnTrueRightOnFalse = probLeft > 0;
    const inKnapsackOnTrue = probIn > 0;

    const numItems = this.items.shape[0];

    const itemBuffer = this.items.bufferSync();
    const oldState = itemBuffer.get(this.cursor.index, 2);
    itemBuffer.set(inKnapsackOnTrue, this.cursor.index, 2);
    this.normalization.visitCounter++;
    itemBuffer.set(
      itemBuffer.get(this.cursor.index, 3) + 1,
      this.cursor.index,
      3
    );
    if (!!oldState === !!inKnapsackOnTrue) {
      this.idleCount++;
    } else {
      this.idleCount = 0;
    }

    let stride;
    if (this.cursor.stride >= 2) {
      stride = this.cursor.stride;
    } else {
      stride = Math.floor(numItems / 2);
      this.cursor.index = Math.floor(numItems / 2);
    }
    stride = Math.floor(stride / 2);
    if (leftOnTrueRightOnFalse) {
      // flip stride sign to go left
      stride = -stride;
    }
    this.cursor.index = this.cursor.index + stride;
    this.cursor.stride = Math.abs(stride);

    return this.isDone();
  }

  value() {
    return tf.tidy(() => {
      const [costPosItems, valuePosItems, inKnapsackPosItems] = tf.unstack(
        this.items,
        1
      );
      const budget = 1; // cost is normalized
      const costInKnapsack = tf.mul(costPosItems, inKnapsackPosItems);
      const countedItems = tf.cumsum(costInKnapsack).lessEqual(budget);
      return tf
        .mul(tf.mul(valuePosItems, inKnapsackPosItems), countedItems)
        .sum()
        .dataSync()[0];
    });
  }

  /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `x` (position of the cart) goes out of bound
   * or when `theta` (angle of the pole) goes out of bound.
   *
   * @returns {bool} Whether the simulation is done.
   */
  isDone() {
    return this.idleCount > this.treeDepth * this.idleThreshold;
  }
}
