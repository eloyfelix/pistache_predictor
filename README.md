# Target predictions C++ REST microservice with ONNX Runtime, RDKit and Pistache

Example of an [histone deacetylase 3](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL1829/) multi-layer perceptron "dummy" model trained in Python using [PyTorch](https://pytorch.org/)(1.4), [exported](https://pytorch.org/docs/stable/onnx.html) to the [ONNX](https://onnx.ai/) format and used with the [ONNX Runtime](https://microsoft.github.io/onnxruntime/) in C++ as a [Pistache](https://github.com/oktal/pistache) powered REST microservice. Molecular fingerprints calculated with [RDKit](https://www.rdkit.org/docs/index.html).

This miniproject only pretends to be an example of how the different components can be glued together in C++ using a model previously trained in Python.

## To run it

### Build the Docker image and run a container

```bash
docker build -t pistache_predictor .
docker run -p9999:9999 pistache_predictor
```

### Query the service:

```bash
curl --request POST -d "Cc1ccccc1-c1ccc(C=NNC(=O)CCCCCC(=O)NO)cc1" http://localhost:9999/predict
```

The trained model is included in the repository but it can be reproduced with the following dataset and script:

- [Dataset](https://github.com/eloyfelix/pistache_predictor/blob/master/training/CHEMBL1829.csv?raw=true) extracted from [ChEMBL](https://www.ebi.ac.uk/chembl).
- Python [script](https://github.com/eloyfelix/pistache_predictor/blob/master/training/train_demo_model.py) to train the model.

## Results comparison

Prediction results of the PyTorch model and of the ONNX exported model running both in Python and C++ ONNX runtimes are compared with the [compare_predictions.py](https://github.com/eloyfelix/pistache_predictor/blob/master/compare_predictions.py) script. Results are the same for all of them.

## LibTorch

The pistache micro-service is also alternatively implemented using the LibTorch C++ backend. It can be used by replacing the Dockerfile, CMakeLists.txt and server.cc files with:

- [Dockerfile_libtorch](https://github.com/eloyfelix/pistache_predictor/blob/master/Dockerfile_libtorch)
- [CMakeLists.txt_libtorch](https://github.com/eloyfelix/pistache_predictor/blob/master/CMakeLists.txt_libtorch)
- [server.cc_libtorch](https://github.com/eloyfelix/pistache_predictor/blob/master/src/server.cc_libtorch)
