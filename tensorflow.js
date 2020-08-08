let tensorflow;
if (!tensorflow) {
  if (!jest) {
    tensorflow = require("@tensorflow/tfjs");
  } else {
    tensorflow = require("@tensorflow/tfjs-node");
  }
}
module.exports = tensorflow;
