// This tests a basic neural network with three logic gates: AND, OR, and NOT
describe('Basic Neural Network', function () {

  // This tests the AND gate
  it("trains an AND gate", function () {

    // Create input and output layers with 2 and 1 neurons, respectively, and project them
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);

    // Train the network to output the correct values for the AND gate
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

    // Test the network with each possible input and assert that the output is correct for each
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // This tests the OR gate
  it("trains an OR gate", function () {

    // Create input and output layers with 2 and 1 neurons, respectively, and project them
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);

    // Train the network to output the correct values for the OR gate
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

    // Test the network with each possible input and assert that the output is correct for each
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // This tests the NOT gate
  it("trains a NOT gate", function () {

    // Create input and output layers with 1 neuron each and project them
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    // Train the network to output the correct values for the NOT gate
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // Test the network with each possible input and assert that the output is correct for each
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

// This tests a Perceptron network's ability to approximate the sine function
describe("Perceptron - SIN", function () {

  // Define the sine function for test purposes
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new Perceptron network with 1 input layer neuron, 12 hidden layer neurons, and 1 output layer neuron
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate a training set consisting of random inputs and their corresponding sine function output
  var trainingSet = [];
  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // Train the network to approximate the sine function
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // Test the network with specific inputs and assert that the output is correct within a margin of error
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  // Test the error of the network and assert that it is less than or equal to the desired error
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

// This tests a Perceptron network's ability to approximate the sine function with cross-validation
describe("Perceptron - SIN - CrossValidate", function () {

  // Define the sine function for test purposes
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Create a new Perceptron network with 1 input layer neuron, 12 hidden layer neurons, and 1 output layer neuron
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  // Generate a training set consisting of random inputs and their corresponding sine function output
  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  // Train the network to approximate the sine function and use cross-validation
  var results = trainer.train(trainingSet, {
    iterations: