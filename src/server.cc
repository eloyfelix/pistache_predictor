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
            module = torch::jit::load("/app/src/demo_model.pt");
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error loading the model" << std::endl;
        }
    }

    void init(size_t thr = 2)
    {
        auto opts = Http::Endpoint::options()
                        .threads(thr);
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
        Routes::Post(router, "/mcp", Routes::bind(&TargetPredictions::mcp, this));
    }

    void mcp(const Rest::Request &request, Http::ResponseWriter response)
    {
        // load mol from body SMILES
        std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(request.body()));

        // calculate fps
        RDKit::SparseIntVect<std::uint32_t> *fp = new RDKit::SparseIntVect<std::uint32_t>(fpSize);
        fp = RDKit::MorganFingerprints::getHashedFingerprint(*mol, 2, fpSize);

        // torch tensor to store query fps and copy the on bits on it
        at::Tensor fpTensor = torch::zeros({1, fpSize});
        for (int i = 1; i < fpSize; ++i)
        {
            if (fp->getVal(i) == 1)
            {
                fpTensor[0][i] = fp->getVal(i);
            }
        }

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(fpTensor);

        // Execute the model and turn its output into a tensor.
        at::Tensor preds = module.forward(inputs).toTensor();
        std::vector<float> predV(preds.data_ptr<float>(), preds.data_ptr<float>() + preds.numel());

        json output;
        output["smiles"] = request.body();
        output["preds"] = predV;

        response.send(Http::Code::Ok, output.dump());
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
