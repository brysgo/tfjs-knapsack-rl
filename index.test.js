import { Knapsack } from "./knapsack";
import { PolicyNetwork } from "./index";
import "./tensorflowMatchers";
import * as tf from "./tensorflow";

jest.mock("./ui", () => ({
  maybeRenderDuringTraining: () => {},
  onGameEnd: () => {},
  setUpUI: () => {},
}));

describe("getLogitsAndActions", () => {
  it("gets and processes the prediction from the input", () => {
    let input, policyNetwork;
    try {
      const knapsack = new Knapsack();
      input = knapsack.getStateTensor();
      policyNetwork = new PolicyNetwork([128]);
    } catch (e) {
      // don't fail because of setup
      console.error(e);
    }
    expect(() => policyNetwork.getLogitsAndActions(input)).not.toThrow();
  });
});

describe("policyNet", () => {
  it("returns the right dimensions for the output", () => {
    let input, policyNetwork, knapsack;
    try {
      knapsack = new Knapsack();
      input = knapsack.getStateTensor();
      policyNetwork = new PolicyNetwork([128]);
    } catch (e) {
      // don't fail because of setup
      console.error(e);
    }
    expect(policyNetwork.policyNet.outputs[0].shape).toEqual([null, 2]);
    expect(policyNetwork.policyNet.predict(input.expandDims()).shape).toEqual([
      1,
      2,
    ]);
  });

  it("can train the network without errors", async () => {
    let input, knapsack, policyNetwork;
    try {
      knapsack = new Knapsack();
      input = knapsack.getStateTensor();
      policyNetwork = new PolicyNetwork([128]);
    } catch (e) {
      // don't fail because of setup
      console.error(e);
    }

    const optimizer = tf.train.adam(0.05);

    await policyNetwork.train(knapsack, optimizer, 0.95, 1, 100);
  });
});
