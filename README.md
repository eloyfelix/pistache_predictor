# Target predictions C++ REST microservice with LibTorch, RDKit and Pistache

Example of an [histone deacetylase 3](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL1829/) multilayer perceptron "dummy" model trained in Python's [PyTorch](https://pytorch.org/)(1.4) and [serialized](https://pytorch.org/tutorials/advanced/cpp_export.html) to be used in C++ as a [Pistache](https://github.com/oktal/pistache) powered REST microservice. Molecular fingerprints calculated with [RDKit](https://www.rdkit.org/docs/index.html).

This miniproject only pretends to be an example of how to glue all the different components in C++ using a model previously trained in Python.

## To run it

### Build the Docker image and run a container

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
