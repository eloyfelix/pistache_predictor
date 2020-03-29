#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>

#include <torch/torch.h>
#include <torch/script.h>

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
            module = torch::jit::load("/app/src/mlp.pt");
        }
        catch (const c10::Error &e)
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
                // Calculate the fingerprints.
                std::unique_ptr<RDKit::SparseIntVect<std::uint32_t>> fp(RDKit::MorganFingerprints::getHashedFingerprint(*mol, 2, fpSize));

                // Copy the ON bits to a new zeros tensor.
                at::Tensor fpTensor = torch::zeros({1, fpSize});
                for (int i = 1; i < fpSize; ++i)
                    if (fp->getVal(i) == 1)
                        fpTensor[0][i] = 1;

                // Create a vector of inputs.
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(fpTensor);

                // Execute the model and turn its output into a tensor.
                at::Tensor preds = module.forward(inputs).toTensor();
                std::vector<float> predVec(preds.data_ptr<float>(), preds.data_ptr<float>() + preds.numel());

                json output;
                output["smiles"] = request.body();
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

    std::shared_ptr<Http::Endpoint> httpEndpoint;
    torch::jit::script::Module module;

    Rest::Router router;
    int fpSize = 2048;
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
