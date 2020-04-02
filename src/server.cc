#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

using namespace Pistache;
using json = nlohmann::json;

namespace Generic
{

void handleReady(const Rest::Request &, Http::ResponseWriter response)
{
    response.send(Http::Code::Ok, "1");
}

} // namespace Generic

class TargetPredictions
{
public:
    TargetPredictions(Address addr) : httpEndpoint(std::make_shared<Http::Endpoint>(addr))
    {
        try
        {
            env = std::move(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

            // initialize session options if needed
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = std::move(Ort::Session(env, "/app/src/mlp.onnx", session_options));

            Ort::AllocatorWithDefaultOptions allocator;

            // only one input node in this model, using index 0 to get its info
            input_node_names.push_back(session.GetInputName(0, allocator));
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            input_node_dims = tensor_info.GetShape();
        }
        catch (...)
        {
            std::cerr << "error loading the model" << std::endl;
        }
    }

    void init(size_t thr = 2)
    {
        auto opts = Http::Endpoint::options().threads(thr);
        httpEndpoint->init(opts);
        setupRoutes();
    }

    void start()
    {
        httpEndpoint->setHandler(router.handler());
        httpEndpoint->serve();
    }

private:
    void setupRoutes()
    {
        using namespace Rest;
        Routes::Get(router, "/ready", Routes::bind(&Generic::handleReady));
        Routes::Post(router, "/predict", Routes::bind(&TargetPredictions::predict, this));
    }

    void predict(const Rest::Request &request, Http::ResponseWriter response)
    {
        try
        {
            std::string smiles(request.body());

            std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
            if (!mol)
            {
                std::cerr << "Cannot create molecule from : '" << smiles << "'" << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Cannot create molecule from : '" + smiles + "'");
            }
            else
            {
                // calc fingerprints
                std::unique_ptr<RDKit::SparseIntVect<std::uint32_t>> fp(RDKit::MorganFingerprints::getHashedFingerprint(*mol, 2, input_node_dims[0]));
                std::vector<float> input_tensor_values(input_node_dims[0], 0);
                for (const auto &iter : fp->getNonzeroElements())
                    input_tensor_values[iter.first] = 1;

                // create input tensor object from data values
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_node_dims[0], input_node_dims.data(), 1);

                // // score model & input tensor, get back output tensor
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
                std::vector<float> predVec(output_tensors.front().GetTensorMutableData<float>(), output_tensors.front().GetTensorMutableData<float>() + output_tensors.size());

                json output;
                output["smiles"] = smiles;
                output["pred"] = predVec;

                response.send(Http::Code::Ok, output.dump());
            }
        }
        catch (RDKit::MolSanitizeException &e)
        {
            std::cerr << e.what() << std::endl;
            response.send(Http::Code::Internal_Server_Error, e.what());
        }
    }

    Ort::Env env{nullptr};
    Ort::Session session{nullptr};

    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names = {"output"};
    std::vector<int64_t> input_node_dims;  

    Rest::Router router;
    std::shared_ptr<Http::Endpoint> httpEndpoint;
};

int main(int argc, char *argv[])
{
    Port port(9080);

    int thr = 2;

    if (argc >= 2)
    {
        port = std::stol(argv[1]);
        if (argc == 3)
            thr = std::stol(argv[2]);
    }

    Address addr(Ipv4::any(), port);

    TargetPredictions tp(addr);

    tp.init(thr);
    std::cout << "TargetPredictions endpoint started" << std::endl;
    tp.start();
}
