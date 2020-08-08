import * as tf from "./tensorflow";
import { pad } from "./utils";
import "./tensorflowMatchers";

describe("utils", () => {
  describe("pad", () => {
    it.tidy(
      "unlike tf implementation, doesn't break when the tensor is empty",
      () => {
        expect(() => tf.pad(tf.ones([0, 3]), [[5, 6]], 12)).toThrow();
        expect(
          pad(
            tf.ones([0, 3]),
            [
              [5, 6],
              [0, 0],
            ],
            12
          )
        ).toEqualTensor(tf.fill([11, 3], 12));
      }
    );
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
