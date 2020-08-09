import * as tf from "./tensorflow";
import { Knapsack } from "./knapsack";
import "./tensorflowMatchers";
import { onesLike } from "@tensorflow/tfjs";

Math.seedrandom("deterministic test results");

describe("knapsack", () => {
  describe("getStateTensor", () => {
    it("has the right dimensions", () => {
      const knapsack = new Knapsack();
      let state;
      expect(() => (state = knapsack.getStateTensor())).not.toThrow();
      expect(state.shape).toEqual([2, 2, 2]);
    });
  });
  describe("value", () => {
    it("returns zero if the knapsack is over", () => {
      const knapsack = new Knapsack();
      knapsack.setRandomState();
      const [costPosItems, valuePosItems, inKnapsackPosItems] = tf.unstack(
        knapsack.items,
        1
      );
      knapsack.items = tf.stack(
        [
          tf.onesLike(costPosItems),
          valuePosItems,
          tf.onesLike(inKnapsackPosItems),
        ],
        1
      );
      expect(knapsack.value()).toBe(0);
    });
    it("returns the value in the knapsack", () => {
      const knapsack = new Knapsack();
      knapsack.setRandomState();
      const [costPosItems, valuePosItems, inKnapsackPosItems] = tf.unstack(
        knapsack.items,
        1
      );
      knapsack.items = tf.stack(
        [
          tf.truncatedNormal(costPosItems.shape),
          valuePosItems,
          inKnapsackPosItems,
        ],
        1
      );
      expect(knapsack.value()).toMatchInlineSnapshot(`0.5881654620170593`);
    });
  });
});

describe("battle tests", () => {
  describe("tf.pad", () => {
    it("doesn't break when the tensor is empty", () => {
      expect(tf.pad(tf.tensor([]), [[5, 6]], 12)).toEqualTensor(
        tf.fill([11], 12)
      );
    });
    it("doesn't break when there is no padding needed", () => {
      expect(tf.pad(tf.tensor([1, 2, 3, 4]), [[0, 0]], 12)).toEqualTensor(
        tf.tensor([1, 2, 3, 4])
      );
    });
    it("works with other dimensions", () => {
      expect(
        tf.pad(
          tf.ones([430, 3]),
          [
            [0, 0],
            [0, 0],
          ],
          0
        )
      ).toEqualTensor(tf.ones([430, 3]));
    });
  });
});
