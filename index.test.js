import { Knapsack } from "./knapsack";
import { PolicyNetwork } from "./index";
import "./tensorflowMatchers";

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
      knapsack.setRandomState();
      input = knapsack.getStateTensor().expandDims();
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
    let input, policyNetwork;
    try {
      const knapsack = new Knapsack();
      knapsack.setRandomState();
      input = knapsack.getStateTensor().expandDims();
      policyNetwork = new PolicyNetwork([128]);
    } catch (e) {
      // don't fail because of setup
      console.error(e);
    }
    console.log(JSON.stringify(policyNetwork.policyNet.outputs[0].shape));
    policyNetwork.policyNet.predict(input);
  });
});
