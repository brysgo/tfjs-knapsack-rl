import * as tf from "@tensorflow/tfjs-node";
import { pad } from "./utils";
import "./tensorflowMatchers";

describe("pad", () => {
  it("works like tf.pad under normal cases", () => {
    tf.tidy(() => {
      const args = [tf.tensor([1, 2, 3, 4]), [[2, 3]], 2];
      expect(tf.pad(...args)).toEqualTensor(pad(...args));
    });
  });
  it("works doesn't break when the tensor is empty", () => {
    tf.tidy(() => {
      expect(pad(tf.tensor([]), [[5, 6]], 12)).toEqualTensor(tf.fill([11], 12));
    });
  });
  it("doesn't break when there is no padding needed", () => {
    tf.tidy(() => {
      expect(pad(tf.tensor([1, 2, 3, 4]), [[0, 0]], 12)).toEqualTensor(
        tf.tensor([1, 2, 3, 4])
      );
    });
  });
});
