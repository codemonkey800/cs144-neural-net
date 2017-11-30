# CS144 Neural Net Project

CS144 Project: Simple neural net that recognizes handwriting.

## Data Sets
Thank you to [Joseph Redmon](https://pjreddie.com) for his
[data sets](https://pjreddie.com/projects/mnist-in-csv).

Inside the repo are a couple of data sets provided by Joseph Redmon. You can
run the neural network by running `build/project` and piping one of the `.csv`
files located in
[data/](https://github.com/codemonkey800/cs144-neural-net/tree/master/data).

The test data set contains 10,000 labeled inputs while the training data set
contains 60,000 labeled inputs.

## Usage

To build the project, run:
```sh
$ make
```

You can also run `make clean` or `make lint`.

The built binary is located at `build/project`. You can get a help message by running:
```sh
$ build/project -h
Usage: build/project [-v|-d|-l] < data/mnist_test.csv

Flags:
  -v - Enable verbose output.
  -d - Dump network weights after training.
  -l - Load network weights from previous training.
```

The neural network uses the file `weights.data` in the current directory for
dumping and loading to and from a file. When verbose is enabled, extra messages
during training and network prediction matching are printed.

## Existing Weights

I've already taken the liberty of generating the weights and saving them to
files for later use. In order to use these weights, you'll have to
copy/move/symlink the files to `weights.data` in your current directory:
```sh
$ cp weights/test.data weights.data
$ build/project -v -l < data/mnist_test.csv
```

This will create a copy of the weights. When the network is run with the `-l`
flag, the weights will be fed into the network and training will be skipped
unless the weights file can't be read or parsed.

The weights can end up being unparseable if some data fails to write to the file or if the
network parameters or test data are changed, but the weights aren't.

## Emoji Index :heart_eyes:
I've recently started using Emojis in a lot of my commit messages, READMEs, and
wherever I can on GitHub. Here's what each emoji I use represents. It may be
subject to change:

- :tada: - Commits that include something so amazing that I have to celebrate :tada:
- :wrench: - Commits that are relatively small to medium in size
- :warning: - Commits that introduce configs or code that break things


## License

MIT License

Copyright (c) 2017 Jeremy Asuncion

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

