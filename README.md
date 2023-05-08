# SAnD Reproduction
1. I provide a key to my Comet ML account, but this is subject to change. You will need to register an account. It is free for academic users. Insert your own API key and user name in the top cell.
1. Download the MIMIC-III Dataset from Physionet.
2. Follow the steps exactly for creating the [MIMIC-III Benchmark](https://github.com/YerevaNN/mimic3-benchmarks) for the task (In the case of this, only In-Hopsital Mortality is required).
3. Connect this dataset to the notebook in place of the reader.
5. Ensure that your execution path is the root of this repository.
6. The notebooks will now be runnable.
7. You will need a GPU with 12GB+ RAM.


If you are using a pre-trained model from a file, it is can be loaded in using the `restore_from_file` method in `utils.trainer`.