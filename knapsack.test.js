import * as tf from "@tensorflow/tfjs-node";
import "./tensorflowMatchers";
import { Knapsack } from "./knapsack";

describe("getStateTensor", () => {
  it("doesn't break", () => {
    const knapsack = new Knapsack();
    expect(() => knapsack.getStateTensor()).not.toThrow();
  });
});

describe("battle tests", () => {
  describe("tf.pad", () => {
    it("doesn't break when the tensor is empty", () => {
      tf.tidy(() => {
        expect(tf.pad(tf.tensor([]), [[5, 6]], 12)).toEqualTensor(
          tf.fill([11], 12)
        );
      });
    });
    it("doesn't break when there is no padding needed", () => {
      tf.tidy(() => {
        expect(tf.pad(tf.tensor([1, 2, 3, 4]), [[0, 0]], 12)).toEqualTensor(
          tf.tensor([1, 2, 3, 4])
        );
      });
    });
    it("works with other dimensions", () => {
      tf.tidy(() => {
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
});
