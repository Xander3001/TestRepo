// Test for a basic neural network
describe('Basic Neural Network', function () {

  // Test for an AND gate
  it("trains an AND gate", function () {

    // Create layers for input and output
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect the layers
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);

    // Define the training set for the AND gate
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
      iterations: 1000, // Number of iterations to train the network
      error: .001 // Acceptable error rate
    });

    // Check the output for each possible input to the AND gate
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // Test for an OR gate
  it("trains an OR gate", function () {

    // Create layers for input and output
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // Connect the layers
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);

    // Define the training set for the OR gate
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
      iterations: 1000, // Number of iterations to train the network
      error: .001 // Acceptable error rate
    });

    // Check the output for each possible input to the OR gate
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // Test for a NOT gate
  it("trains a NOT gate", function () {

    // Create layers for input and output
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    // Connect the layers
    inputLayer.project(outputLayer);

    // Create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // Create a trainer for the network
    var trainer = new Trainer(network);

    // Define the training set for the NOT gate
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    // Train the network with the training set
    trainer.train(trainingSet, {
      iterations: 1000, // Number of iterations to train the network
      error: .001 // Acceptable error rate
    });

    // Check the output for each possible input to the NOT gate
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

// Test for a Perceptron with SIN function
describe("Perceptron - SIN", function () {

  // Custom function for the sine of x shifted and scaled to be between 0 and 1
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Set up a Perceptron with input layer size 1, hidden layer size 12, and output layer size 1
  var sinNetwork = new Perceptron(1, 12, 1);

  // Create a trainer for the network
  var trainer = new Trainer(sinNetwork);

  // Create a training set with 800 random inputs between 0 and 2*pi and their corresponding outputs
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // Train the network with the training set
  var results = trainer.train(trainingSet, {
    iterations: 2000, // Number of iterations to train the network
    log: false, // Whether to log training output
    error: 1e-6, // Acceptable error rate
    cost: Trainer.cost.MSE, // Mean Squared Error cost function
  });

  // Test the network with specific inputs and their expected outputs
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // Check that the output is within a certain error range of the expected output
        // Acceptable error rate is 0.5 * 10^-decimal, where decimal is the number of decimal places to check
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  // Check the final error rate for the training of the network
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

// Test for a Perceptron with SIN function using cross-validation
describe("Perceptron - SIN - CrossValidate", function () {

  // Custom function for the sine of x shifted and scaled to be between 0 and 1
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // Set up a Perceptron with input layer size 1, hidden layer size 12, and output layer size 1
  var sinNetwork = new Perceptron(1, 12, 1);

  // Create a trainer for the network
  var trainer = new Trainer(sinNetwork);

  // Create a training set with 800 random inputs between 0 and 2*pi and their corresponding outputs