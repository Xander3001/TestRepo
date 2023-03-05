/**
 * This test suite checks the functionality of different neural network models using a set of specified tasks.
 */

describe('Basic Neural Network', function () {
  
  /**
   * The following test case trains a network for an AND gate using a Layer and a Trainer.
   * It then checks the outputs of the network for all possible input combinations of an AND gate.
   */
  it("trains an AND gate", function () {

    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);

    var trainingSet = [{
      input: [0, 0],
      output: [0]
    }, {
      input: [0, 1],
      output: [0]
    }, {
      input: [1, 0],
      output: [0]
    }, {
      input: [1, 1],
      output: [1]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  /**
   * The following test case trains a network for an OR gate using a Layer and a Trainer.
   * It then checks the outputs of the network for all possible input combinations of an OR gate.
   */
  it("trains an OR gate", function () {

    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);

    var trainingSet = [{
      input: [0, 0],
      output: [0]
    }, {
      input: [0, 1],
      output: [1]
    }, {
      input: [1, 0],
      output: [1]
    }, {
      input: [1, 1],
      output: [1]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  /**
   * The following test case trains a network for a NOT gate using a Layer and a Trainer.
   * It then checks the outputs of the network for the possible inputs of a NOT gate.
   */
  it("trains a NOT gate", function () {

    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

/**
 * The following test case trains a Perceptron for a sine function using a Trainer.
 * It generates random data for the input of the Perceptron and compares the outputs with the expected outputs.
 * The error of training is also checked.
 */
describe("Perceptron - SIN", function () {
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

/**
 * The following test case trains a Perceptron for a sine function using a Trainer with cross-validation.
 * It generates random data for the input of the Perceptron and compares the outputs with the expected outputs.
 * The error of training is also checked.
 */
describe("Perceptron - SIN - CrossValidate", function () {

  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
    crossValidate: {
      testSize: .3,
      testError: 1e-6
    }
  });

  var test0 = sinNetwork.activate([0])[0];
  var expected0 = mySin(0);
  it("input: [0] output: " + test0 + ", expected: " + expected0, function () {
    assert.isAtMost(Math.abs(test0 - expected0), .035, "[0] did not output " + expected0);
  });

  var test05PI = sinNetwork.activate([.5 * Math.PI])[0];
  var expected05PI = mySin(.5 * Math.PI);
  it("input: [0.5*Math.PI] output: " + test05PI + ", expected: " + expected05PI, function () {
    assert.isAtMost(Math.abs(test05PI - expected05PI), .035, "[0.5*Math.PI] did not output " + expected05