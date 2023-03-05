// This code tests the functionality of various neural network models using different datasets.
// The first section tests a basic neural network on AND, OR, and NOT gates.
// The second section tests a perceptron on the sine function, both with and without cross-validation.
// The third section tests a long short-term memory network on sequence recall.

describe('Basic Neural Network', function () {

  // Test the network's ability to learn the AND gate
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

    // Test each input to ensure correct output
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // Test the network's ability to learn the OR gate
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

    // Test each input to ensure correct output
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // Test the network's ability to learn the NOT gate
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

    // Test each input to ensure correct output
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});


describe("Perceptron - SIN", function () {

  // Define a function to calculate the sine function and normalize the output to between 0 and 1
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new perceptron with 1 input layer, 12 hidden neurons, and 1 output layer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate a training set of 800 inputs and outputs based on the sine function
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // Train the network using the training set and set training options
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // Test the network's output for specific input values and expected output values
  [0, .5 * Math.PI, 2].forEach(function (x) {
    var y = mySin(x);
    it("should return value around " + y + " when [" + x + "] is on input", function () {
      // Define a margin of error for the output
      // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
      // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
      assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
    });
  });

  // Check the final error rate after training
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

describe("Perceptron - SIN - CrossValidate", function () {

  // Define a function to calculate the sine function and normalize the output to between 0 and 1
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new perceptron with 1 input layer, 12 hidden neurons, and 1 output layer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate a training set of 800 inputs and outputs based on the sine function
  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  // Train the network using the training set and set training and validation options
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

  // Test the network's output for specific input values and expected output values
  var test0 = sinNetwork.activate([0])[0];
  var expected0 = mySin(0);
  it("input: [0] output: " + test0 + ", expected: " + expected0, function () {
    assert.isAtMost(Math.abs(test0 - expected0), .035,