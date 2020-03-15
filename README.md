# Target predictions C++ REST microservice with libtorch, RDKit and Pistache

Example of an histone deacetylase 3 multilayer perceptron model trained in Python's [PyTorch](https://pytorch.org/)(1.4) and [serialized](https://pytorch.org/tutorials/advanced/cpp_export.html) to be used in C++ as a [Pistache](https://github.com/oktal/pistache) powered REST microservice. This only pretends to be an example of how to glue all components together.

## To run it

### Build the Docker image and run it

The included Dockerfile downloads and compiles [RDKit](https://www.rdkit.org/docs/index.html) along with all other dependecies:

```bash
docker build -t pistache_predictor .
docker run -p9999:9999 pistache_predictor
```

### Query the service:

```bash
curl --request POST -d 'Cc1ccccc1-c1ccc(C=NNC(=O)CCCCCC(=O)NO)cc1' http://localhost:9999/predict
```

The trained model is included in the repository but it can be reproduced with the following dataset and script:

- [Dataset](https://github.com/eloyfelix/pistache_predictor/blob/master/src/CHEMBL1829.csv?raw=true) extracted from [ChEMBL](https://www.ebi.ac.uk/chembl).
- Python [script](https://github.com/eloyfelix/pistache_predictor/blob/master/src/train_demo_model.py) to train the model.
