/* 
This file contains tests for a basic neural network, a perceptron for computing the sine function, and an LSTM 
for discrete sequence recall. Each section has its own describe block, with tests specified in individual it blocks.
The basic neural network tests train AND, OR, and NOT gates, and then tests their outputs. The perceptron tests 
the accuracy of predicted values for the sine function. The LSTM is trained to recall a specific sequence, and 
tests whether the network is accurately predicting the next value in the sequence. 
*/ 

describe('Basic Neural Network', function () {
  // Trains an AND gate
  it("trains an AND gate", function () {
    // Creates input and output layers, and adds a connection between them.
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);
    // Creates the network and a trainer for it, using the layers created above.
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });
    var trainer = new Trainer(network);
    // Defines the training set for the AND gate, with inputs, and expected outputs.
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
    // Trains the network on the AND gate and defines the number of iterations and desired error rate.
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });
    // Case-specific tests to ensure that the network is properly classifying inputs.
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");
    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");
    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");
    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });
  // Trains an OR gate
  it("trains an OR gate", function () {
    // Creates input and output layers, and adds a connection between them.
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);
    // Creates the network and a trainer for it, using the layers created above.
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });
    var trainer = new Trainer(network);
    // Defines the training set for the OR gate, with inputs, and expected outputs.
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
    // Trains the network on the OR gate and defines the number of iterations and desired error rate.
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });
    // Case-specific tests to ensure that the network is properly classifying inputs.
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");
    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");
    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");
    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });
  // Trains a NOT gate
  it("trains a NOT gate", function () {
    // Creates input and output layers, and adds a connection between them.
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);
    inputLayer.project(outputLayer);
    // Creates the network and a trainer for it, using the layers created above.
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });
    var trainer = new Trainer(network);
    // Defines the training set for the NOT gate, with inputs, and expected outputs.
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];
    // Trains the network on the NOT gate and defines the number of iterations and desired error rate.
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });
    // Case-specific tests to ensure that the network is properly classifying inputs.
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");
    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

describe("Perceptron - SIN", function () {
  // Function to compute value of sine function
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };
  // Creates a new perceptron with input layer size 1, hidden layer size 12, and output layer size 1.
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);
  // Creates a random set of training data of size 800, with inputs on the interval [0, 2pi], and computes the 
  // sine of these inputs as the expected output.
  var trainingSet = [];
  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }
  // Trains the perceptron using the trainingSet data.
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // Tests the predicted output of the trained perceptron on known input values in the range [0,0.5pi,2].
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // Defines an acceptable level of error between the expected output and the predicted output.
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });
  // Computes the training error rate
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    // Tests that the error is less than or equal to 0.001.
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

describe("Perceptron - SIN - CrossValidate", function () {
  // Function to compute value of sine function
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };
  // Creates a new perceptron with input layer size 1, hidden layer size 12, and output layer size 1.
  var sinNetwork = new Perceptron(1,