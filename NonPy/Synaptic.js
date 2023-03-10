

describe('Basic Neural Network', function () {

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
    assert.isAtMost(Math.abs(test05PI - expected05PI), .035, "[0.5*Math.PI] did not output " + expected05PI);
  });

  var test2 = sinNetwork.activate([2])[0];
  var expected2 = mySin(2);
  it("input: [2] output: " + test2 + ", expected: " + expected2, function () {
    var eq = equalWithError(test2, expected2, .035);
    assert.equal(eq, true, "[2] did not output " + expected2);
  });

  var errorResult = results.error;
  it("CrossValidation error: " + errorResult, function () {
    var lessThanOrEqualError = errorResult <= .001;
    assert.equal(lessThanOrEqualError, true, "CrossValidation error not less than or equal to desired error.");
  });
});

describe("LSTM - Discrete Sequence Recall", function () {
  var targets = [2, 4];
  var distractors = [3, 5];
  var prompts = [0, 1];
  var length = 9;

  var lstm = new LSTM(5, 3, 2);
  var trainer = new Trainer(lstm);

  trainer.DSR({
    targets: targets,
    distractors: distractors,
    prompts: prompts,
    length: length,
    rate: .17,
    iterations: 250000
  });

  var symbols = targets.length + distractors.length + prompts.length;
  var sequence = [],
    indexes = [],
    positions = [];
  var sequenceLength = length - prompts.length;

  for (i = 0; i < sequenceLength; i++) {
    var any = Math.random() * distractors.length | 0;
    sequence.push(distractors[any]);
  }
  indexes = [], positions = [];
  for (i = 0; i < prompts.length; i++) {
    indexes.push(Math.random() * targets.length | 0);
    positions.push(noRepeat(sequenceLength, positions));
  }
  positions = positions.sort();
  for (i = 0; i < prompts.length; i++) {
    sequence[positions[i]] = targets[indexes[i]];
    sequence.push(prompts[i]);
  }

  var check = function (which) {
    // generate input from sequence
    var input = [];
    for (let j = 0; j < symbols; j++)
      input[j] = 0;
    input[sequence[which]] = 1;

    // generate target output
    var output = [];
    for (let j = 0; j < targets.length; j++)
      output[j] = 0;

    if (which >= sequenceLength) {
      var index = which - sequenceLength;
      output[indexes[index]] = 1;
    }

    // check result
    var prediction = lstm.activate(input);
    return {
      prediction: prediction,
      output: output
    };
  };

  var value = function (array) {
    var max = .5;
    var res = -1;
    for (var i in array)
      if (array[i] > max) {
        max = array[i];
        res = i;
      }
    return res == -1 ? '-' : targets[res];
  };

  it("targets: " + targets, function () {
    assert(true);
  });
  it("distractors: " + distractors, function () {
    assert(true);
  });
  it("prompts: " + prompts, function () {
    assert(true);
  });
  it("length: " + length + "\n", function () {
    assert(true);
  });

  for (var i = 0; i < length; i++) {
    var test = check(i);
    it((i + 1) + ") input: " + sequence[i] + " output: " + value(test.prediction),
      function () {
        var ok = equal(test.prediction, test.output);
        assert(ok);
      });
  }
});



