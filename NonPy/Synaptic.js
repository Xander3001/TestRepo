// This describes a suite of tests for a basic neural network
describe('Basic Neural Network', function () {

  // This tests whether the network can learn to perform an AND gate
  it("trains an AND gate", function () {

    // Create a new input layer with 2 nodes and an output layer with 1 node
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    // Create a new network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a new trainer using the network
    var trainer = new Trainer(network);

    // Define a set of training data for the AND gate
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

    // Train the network with the training data for 1000 iterations or until the error is below .001
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and check the outputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // This tests whether the network can learn to perform an OR gate
  it("trains an OR gate", function () {

    // Create a new input layer with 2 nodes and an output layer with 1 node
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    // Create a new network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a new trainer using the network
    var trainer = new Trainer(network);

    // Define a set of training data for the OR gate
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

    // Train the network with the training data for 1000 iterations or until the error is below .001
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and check the outputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // This tests whether the network can learn to perform a NOT gate
  it("trains a NOT gate", function () {

    // Create a new input layer with 1 node and an output layer with 1 node
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    // Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    // Create a new network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a new trainer using the network
    var trainer = new Trainer(network);

    // Define a set of training data for the NOT gate
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    // Train the network with the training data for 1000 iterations or until the error is below .001
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with different inputs and check the outputs
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

// This describes a suite of tests for a perceptron trained to approximate the sin function
describe("Perceptron - SIN", function () {

  // Define the sin function to be approximated
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new perceptron with 1 input node, 12 hidden nodes, and 1 output node
  var sinNetwork = new Perceptron(1, 12, 1);

  // Create a new trainer using the perceptron
  var trainer = new Trainer(sinNetwork);

  // Generate a set of training data for the sin function
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // Train the perceptron with the training data for 2000 iterations or until the error is below 1e-6
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // Test the perceptron with different inputs and check that the outputs are within a reasonable margin of error
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // Set the margin of error
        // abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  // Check that the error of the perceptron is below 0.001
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

// This describes a suite of tests for a perceptron trained to approximate the sin function using cross-validation
describe("Perceptron - SIN - CrossValidate", function () {

  // Define the sin function to be approximated
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new perceptron with 1 input node, 12 hidden nodes, and 1 output node
  var sinNetwork = new