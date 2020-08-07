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

import * as tf from "@tensorflow/tfjs";

/**
 * Knapsack system simulator.
 *
 * The state is a tensor of shape [2, 2, 2]:
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

    // Threshold values, beyond which a simulation will be marked as failed.
    this.xThreshold = 2.4;
    this.thetaThreshold = (12 / 360) * 2 * Math.PI;

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
    // [numberOfItems, (value,cost,inKnapsack)]
    tf.dispose(this.items);
    this.items = tf.tidy(() =>
      tf.concat(
        [
          tf.randomUniform([numItems, 2]),
          tf.randomUniform([numItems, 1]).greater(tf.scalar(0.5)),
        ],
        1
      )
    );
    this.cursor = 0;
  }

  /**
   * Get current state as a tf.Tensor of shape [2, 2, 2].
   * [# batches, left/right, in/out, cost/value)]
   *
   */
  getStateTensor() {
    return tf.tidy(() => {
      return tf.stack(
        [this.items.slice(0, this.cursor), this.items.slice(this.cursor)].map(
          (itemsPos) => {
            const [
              valuePosItems,
              costPosItems,
              inKnapsackPosItems,
            ] = tf.unstack(itemsPos, 1);
            const valueCostPos = tf.stack([valuePosItems, costPosItems]);
            return tf.stack(
              tf.mul(valueCostPos, inKnapsackPosItems),
              tf.mul(valueCostPos, inKnapsackPosItems.sub(tf.scalar(1)))
            );
          }
        )
      );
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
  update([probLeft, flipProb]) {
    const leftOnTrueRightOnFalse = probLeft > 0;
    const flipOnTrue = flipProb > 0;

    const numItems = this.items.shape[0];

    if (flipOnTrue) {
      const flipMask = tf.zerosLike(this.items);
      flipMask.bufferSync().set(true, this.cursor);
      const flippedItems = tf.logicalXor(this.items, flipMask);
      tf.dispose(this.items);
      this.items = flippedItems;
    }

    if (leftOnTrueRightOnFalse) {
      // move right by half the list
      this.cursor = Math.floor(this.cursor / 2);
    } else {
      // move left by half the list
      this.cursor = Math.floor(this.cursor / 2);
      const spaceToEnd = numItems - this.cursor;
      this.cursor = this.cursor + Math.floor(spaceToEnd / 2);
    }

    return this.isDone();
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
    return (
      this.x < -this.xThreshold ||
      this.x > this.xThreshold ||
      this.theta < -this.thetaThreshold ||
      this.theta > this.thetaThreshold
    );
  }
}
