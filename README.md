# SAnD Reproduction
1. Install conda
2. I provide a key to my Comet ML account, but this is subject to change. You will need to register an account. It is free for academic users. Insert your own API key and user name in the top cell.
3. Download the MIMIC-III Dataset from Physionet.
4. Follow the steps exactly for creating the [MIMIC-III Benchmark](https://github.com/YerevaNN/mimic3-benchmarks) for the task (In the case of this, only In-Hopsital Mortality is required). It is recommended to make a Conda enviornment from the instructions here.
5. Connect this dataset to the notebook in place of the reader.
6. Ensure that your execution path is the root of this repository.
7. `conda env create -f environment.yml` to recreate the enviornment 
8. The notebooks will now be runnable.
9. You will need a GPU with ~8GB RAM.


If you are using a pre-trained model from a file, it is can be loaded in using the `restore_from_file` method in `utils.trainer.NeuralNetworkClassifier`.
The models are located in `train/models`.