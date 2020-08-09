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
    this.idleThreshold = 1;

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
          tf
            .randomUniform([numItems, 1])
            .greater(tf.scalar(0.5))
            .cast("float32"),
        ],
        1
      )
    );
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
    tf.dispose(this.lastStateTensor);
    this.lastStateTensor = tf.tidy(() => {
      return tf
        .stack(
          [
            pad(this.items.slice(0, this.cursor.index), [
              [0, this.items.shape[0] - this.cursor.index],
              [0, 0],
            ]),
            pad(this.items.slice(this.cursor.index), [
              [this.cursor.index, 0],
              [0, 0],
            ]),
          ].map((itemsPos) => {
            const [
              costPosItems,
              valuePosItems,
              inKnapsackPosItems,
            ] = tf.unstack(itemsPos, 1);
            const valueCostPos = tf.stack([costPosItems, valuePosItems]);
            return tf.stack([
              tf.mul(valueCostPos, inKnapsackPosItems),
              tf.mul(valueCostPos, tf.scalar(1).sub(inKnapsackPosItems)),
            ]);
          })
        )
        .sum(-1);
    });
    return this.lastStateTensor;
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
      this.idleCount = 0;
      const itemBuffer = this.items.bufferSync();
      const newState = !itemBuffer.get(this.cursor.index, 2);
      itemBuffer.set(newState, this.cursor.index, 2);
    } else {
      this.idleCount++;
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
      const isOver = tf
        .mul(costPosItems, inKnapsackPosItems)
        .sum()
        .greater(1)
        .dataSync()[0];
      if (!isOver) {
        return 0;
      }
      return tf.mul(valuePosItems, inKnapsackPosItems).sum().dataSync()[0];
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
