//This code uses a testing framework to test the functionality of neural network models for the AND, OR, and NOT gates, as well as for predicting a discrete sequence recall.

//Test the basic neural network for the AND gate
describe('Basic Neural Network', function () {

  //Train the network for the AND gate
  it("trains an AND gate", function () {
  
    //Create the input and output layers for the network
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    //Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    //Create the network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    //Create the trainer for the network
    var trainer = new Trainer(network);

    //Set up the training set for the AND gate
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

    //Train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    //Test the trained network with various inputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  //Test the basic neural network for the OR gate
  it("trains an OR gate", function () {

    //Create the input and output layers for the network
    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    //Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    //Create the network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    //Create the trainer for the network
    var trainer = new Trainer(network);

    //Set up the training set for the OR gate
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

    //Train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    //Test the trained network with various inputs
    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  //Test the basic neural network for the NOT gate
  it("trains a NOT gate", function () {

    //Create the input and output layers for the network
    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    //Connect the input layer to the output layer
    inputLayer.project(outputLayer);

    //Create the network using the input and output layers
    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    //Create the trainer for the network
    var trainer = new Trainer(network);
    
    //Set up the training set for the NOT gate
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    //Train the network using the training set
    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    //Test the trained network with various inputs
    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

//Test the perceptron for predicting a sin function
describe("Perceptron - SIN", function () {

  //Define the sin function to be predicted
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  //Create the perceptron and trainer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  //Set up the training set
  var trainingSet = [];
  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  //Train the perceptron network on the training set
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  //Test the trained network with various inputs and expected outputs
  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  //Test the sin error of the trained network against a desired error threshold
  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

//Test the perceptron for predicting a sin function with cross-validation
describe("Perceptron - SIN - CrossValidate", function () {

  //Define the sin function to be predicted
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  //Create the perceptron and trainer
  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  //Set up the training set
  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  //Train the perceptron network on the training set with cross-validation
  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-