/**
 * Tests for a basic neural network using different gate training sets: AND, OR, and NOT.
 */
describe('Basic Neural Network', function () {
  
  /**
   * Tests if the neural network can successfully train for an AND gate.
   */
  it("trains an AND gate", function () {
    // Define input and output layers
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect input to output layer
    inputLayer.project(outputLayer);

    // Set up network with input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Set up trainer with the network
    var trainer = new Trainer(network);

    // Set up training set with input and expected output for AND gate
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

    // Train the network with the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and expected outputs for AND gate
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
   * Tests if the neural network can successfully train for an OR gate.
   */
  it("trains an OR gate", function () {
    // Define input and output layers
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect input to output layer
    inputLayer.project(outputLayer);

    // Set up network with input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Set up trainer with the network
    var trainer = new Trainer(network);

    // Set up training set with input and expected output for OR gate
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

    // Train the network with the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and expected outputs for OR gate
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
   * Tests if the neural network can successfully train for a NOT gate.
   */
  it("trains a NOT gate", function () {
    // Define input and output layers
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    // Connect input to output layer
    inputLayer.project(outputLayer);

    // Set up network with input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Set up trainer with the network
    var trainer = new Trainer(network);

    // Set up training set with input and expected output for NOT gate
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    // Train the network with the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and expected outputs for NOT gate
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

/**
 * Tests for a Perceptron model to learn the sine function and cross validate the prediction accuracy.
 */
describe("Perceptron - SIN", function () {
  // Helper function to define the sine function output
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Set up Perceptron model with 1 input layer, 12 hidden neurons, and 1 output layer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate random training set of 800 input and output pairs for sine function
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // Train the model with the sine function training set
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // Test the model with specific inputs for sine function and expected outputs
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  // Check if the error rate of the model is within an acceptable range
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

/**
 * Tests for a Perceptron model to learn the sine function with cross validation.
 */
describe("Perceptron - SIN - CrossValidate", function () {
  // Helper function to define the sine function output
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Set up Perceptron model with 1 input layer, 12 hidden neurons, and 1 output layer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate random training set of 800 input and output pairs for sine function
  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue