# Target prediction C++ REST microservice with ONNX Runtime, RDKit and Pistache

[ChEMBL Multitask Model](https://github.com/chembl/chembl_multitask_model/) C++ REST microservice with [ONNX Runtime](https://microsoft.github.io/onnxruntime/) and [Pistache](https://github.com/pistacheio/pistache). Molecular fingerprints calculated with [RDKit](https://www.rdkit.org/docs/index.html).

## To run it

### Build the Docker image and run a container

```bash
docker-compose up -d
```

### Query the service:

```bash
curl --request POST -d "Cc1ccccc1-c1ccc(C=NNC(=O)CCCCCC(=O)NO)cc1" http://localhost:9080/predict
```

### To stop and delete the running container

```bash
docker-compose down
```
