#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "DataFormats/DigiClusterCount.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {
  std::atomic<int> allEvents = 0;
  std::atomic<int> goodEvents = 0;

  std::mutex sumTrackDifferenceMutex;
  float sumTrackDifference = 0;
}  // namespace

class CountValidator : public edm::EDProducer {
public:
  explicit CountValidator(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

  edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
};

CountValidator::CountValidator(edm::ProductRegistry& reg)
    : digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
      digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()) {}

void CountValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::stringstream ss;
  bool ok = true;

  ss << "Event " << iEvent.eventID() << " ";

  {
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextProduce ctx{pdigis};
    auto const& count = iEvent.get(digiClusterCountToken_);
    auto const& digis = ctx.get(iEvent, digiToken_);
    auto const& clusters = ctx.get(iEvent, clusterToken_);

    if (digis.nModules() != count.nModules()) {
      ss << "\n N(modules) is " << digis.nModules() << " expected " << count.nModules();
      ok = false;
    }
    if (digis.nDigis() != count.nDigis()) {
      ss << "\n N(digis) is " << digis.nDigis() << " expected " << count.nDigis();
      ok = false;
    }
    if (clusters.nClusters() != count.nClusters()) {
      ss << "\n N(clusters) is " << clusters.nClusters() << " expected " << count.nClusters();
      ok = false;
    }
  }

  ++allEvents;
  if (ok) {
    ++goodEvents;
  } else {
    std::cout << ss.str() << std::endl;
  }
}

void CountValidator::endJob() {
  if (allEvents == goodEvents) {
    std::cout << "CountValidator: all " << allEvents << " events passed validation\n";
    if (sumTrackDifference != 0.f) {
      std::cout << " Average relative track difference " << sumTrackDifference / allEvents.load()
                << " (all within tolerance)\n";
    }
  } else {
    std::cout << "CountValidator: " << (allEvents - goodEvents) << " events failed validation (see details above)\n";
    throw std::runtime_error("CountValidator failed");
  }
}

DEFINE_FWK_MODULE(CountValidator);
