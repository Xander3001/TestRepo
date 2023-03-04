// This code contains a series of unit tests for different types of neural networks.
// The first set of tests trains basic neural networks to act as AND, OR and NOT gates.
// The second set of tests trains a Perceptron network to approximate the Sin function.
// The third set of tests trains a Long Short-Term Memory (LSTM) network to recognize and recall discrete sequences of symbols.

describe('Basic Neural Network', function () {

  // trains the network to act as an AND gate
  it("trains an AND gate", function () {

    // create input and output layers
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // connect input and output layers together
    inputLayer.project(outputLayer);

    // create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // create a trainer for the network
    var trainer = new Trainer(network);

    // define the training set
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

    // train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // test the network against various inputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // trains the network to act as an OR gate
  it("trains an OR gate", function () {

    // create input and output layers
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    // connect input and output layers together
    inputLayer.project(outputLayer);

    // create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // create a trainer for the network
    var trainer = new Trainer(network);

    // define the training set
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

    // train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // test the network against various inputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  // trains the network to act as a NOT gate
  it("trains a NOT gate", function () {

    // create input and output layers
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    // connect input and output layers together
    inputLayer.project(outputLayer);

    // create a network with the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    // create a trainer for the network
    var trainer = new Trainer(network);

    // define the training set
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    // train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    // test the network against various inputs
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

describe("Perceptron - SIN", function () {

  // define a helper function to approximate the sin function
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // create a Perceptron network with one input, 12 hidden neurons, and one output
  var sinNetwork = new Perceptron(1, 12, 1);

  // create a trainer for the network
  var trainer = new Trainer(sinNetwork);

  // create a training set of random values
  var trainingSet = [];
  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  // train the network using the training set
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  // test the network against various inputs
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  // check the error rate of the network against the desired error rate
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

describe("Perceptron - SIN - CrossValidate", function () {

  // define a helper function to approximate the sin function
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  // create a Perceptron network with one input, 12 hidden neurons, and one output
  var sinNetwork = new Perceptron(1, 12, 1);

  // create a trainer for the network
  var trainer = new Trainer(sinNetwork);

  // create a training set of random values
  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  // train the network using the training set with cross-validation
  var results = trainer.train(trainingSet, {
    iterations: 200