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
 * In the control-theory sense, there are four state variables in this system:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action:
 *
 *   - leftward or rightward force.
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

    const numItems = this.itemRange.min + Math.random() * this.itemRange.max;

    // [numberOfItems, (value,cost,inKnapsack)]
    this.items = tf.tidy(() =>
      tf.concat(
        [
          tf.randomUniform([numItems, 2]),
          tf.randomUniform([numItems, 1], null, null, "bool"),
        ],
        1
      )
    );
    this.cursor = 0;
  }

  /**
   * Get current state as a tf.Tensor of shape [2, 2, 3].
   * [# batches, in/out, left/right, (cost, value, count)]
   *
   * *current* is a special case where spacial dimensions become quantities
   */
  getStateTensor() {
    return tf.tensor3d([
      // in
      [
        // left
        [this.costInLeft, this.valueInLeft, this.countInLeft],
        // right
        [this.costInRight, this.valueInRight, this.countInRight],
      ],
      // out
      [
        // left
        [this.costOutLeft, this.valueOutLeft, this.countOutLeft],
        // right
        [this.costOutRight, this.valueOutRight, this.countOutRight],
      ],
    ]);
  }

  /**
   * Update the knapsack system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  update([probLeft, flipProb]) {
    const leftOnTrueRightOnFalse = probLeft > 0;
    const flipOnTrue = flipProb > 0;

    this.numItemsRemaining = this.allItems.length - this.enclosedItems.length;
    this.numItemsEnclosed = this.enclosedItems.length;
    this.valueEnclosed = 0;
    this.costEnclosed = 0;
    this.enclosedItems.forEach((enclosedItem) => {
      this.costEnclosed -= enclosedItem.cost;
      this.valueEnclosed += enclosedItem.value;
    });
    // old code vvv
    const force = action > 0 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(this.theta);
    const sinTheta = Math.sin(this.theta);

    const temp =
      (force + this.poleMoment * this.thetaDot * this.thetaDot * sinTheta) /
      this.totalMass;
    const thetaAcc =
      (this.gravity * sinTheta - cosTheta * temp) /
      (this.length *
        (4 / 3 - (this.massPole * cosTheta * cosTheta) / this.totalMass));
    const xAcc =
      temp - (this.poleMoment * thetaAcc * cosTheta) / this.totalMass;

    // Update the four state variables, using Euler's metohd.
    this.x += this.tau * this.xDot;
    this.xDot += this.tau * xAcc;
    this.theta += this.tau * this.thetaDot;
    this.thetaDot += this.tau * thetaAcc;

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
