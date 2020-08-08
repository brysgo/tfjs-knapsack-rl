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
      input = knapsack.getStateTensor();
      policyNetwork = new PolicyNetwork([128]);
    } catch (e) {
      // don't fail because of setup
      console.error(e);
    }
    expect(() => policyNetwork.getLogitsAndActions(input)).not.toThrow();
  });
});
