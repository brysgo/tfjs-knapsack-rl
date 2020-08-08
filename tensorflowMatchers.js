import * as tf from "./tensorflow";
import diff from "jest-diff";

it.tidy = (...args) =>
  tf.tidy(() => {
    it(...args);
  });

expect.extend({
  toEqualTensor(received, expected) {
    const options = {
      comment: "Match tensorflow tensors",
      isNot: this.isNot,
      promise: this.promise,
    };

    let pass;
    let error;
    try {
      pass = received.sub(expected).sum().arraySync() === 0;
    } catch (e) {
      pass = false;
      error = e;
    }

    const message = pass
      ? () =>
          this.utils.matcherHint(
            "toEqualTensor",
            undefined,
            undefined,
            options
          ) +
          "\n\n" +
          `Expected: not ${this.utils.printExpected(expected.arraySync())}\n` +
          `Received: ${this.utils.printReceived(received.arraySync())}`
      : () => {
          if (error) console.error(error);
          const diffString = diff(expected.arraySync(), received.arraySync(), {
            expand: this.expand,
          });
          return (
            this.utils.matcherHint(
              "toEqualTensor",
              undefined,
              undefined,
              options
            ) +
            "\n\n" +
            (diffString && diffString.includes("- Expect")
              ? `Difference:\n\n${diffString}`
              : `Expected: ${this.utils.printExpected(
                  expected.arraySync()
                )}\n` +
                `Received: ${this.utils.printReceived(received.arraySync())}`)
          );
        };

    return { actual: received, message, pass };
  },
});
