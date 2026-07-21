#include <Oa/Core/Thread.h>
#include <Oa/Core/Log.h>

#include <condition_variable>
#include <mutex>

#ifdef OA_PLATFORM_LINUX
#include <pthread.h>
#include <sched.h>
#endif

static bool PinThreadToCore([[maybe_unused]] OaI32 InCoreId) {
#ifdef OA_PLATFORM_LINUX
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(InCoreId, &cpuset);
	return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#else
	return false;
#endif
}

struct OaThreadPool::State {
	OaVec<OaSharedPtr<OaChannel<Job>>> Queues;
	std::atomic<OaI32> NextWorker{0};
	std::atomic<OaI32> WorkersRemaining{0};
	std::atomic<bool> Running{false};
	std::atomic<bool> DrainOnStop{false};
	std::mutex FinishedMutex;
	std::condition_variable Finished;
};

void OaThreadPool::WorkerLoop(
	OaSharedPtr<State> InState,
	OaI32 InWorkerId,
	OaI32 InCoreId,
	OaBool InPinToCore)
{
	if (InPinToCore && InCoreId >= 0) {
		PinThreadToCore(InCoreId);
	}

	const OaI32 numQueues = static_cast<OaI32>(InState->Queues.Size());
	auto& ownQueue = InState->Queues[InWorkerId];
	auto executeOrCancel = [&InState](Job& InJob) {
		if (InState->Running.load(std::memory_order_acquire)
			|| InState->DrainOnStop.load(std::memory_order_acquire)) {
			if (InJob.Run) InJob.Run();
		} else if (InJob.Cancel) {
			InJob.Cancel();
		}
	};

	while (InState->Running.load(std::memory_order_acquire)) {
		// Try own queue first (non-blocking to allow stealing)
		auto job = ownQueue->TryRecv();
		if (job) {
			executeOrCancel(*job);
			continue;
		}

		// Work stealing: try sibling queues
		bool stolen = false;
		for (OaI32 i = 1; i < numQueues; ++i) {
			OaI32 target = (InWorkerId + i) % numQueues;
			auto stealJob = InState->Queues[target]->TryRecv();
			if (stealJob) {
				executeOrCancel(*stealJob);
				stolen = true;
				break;
			}
		}

		if (!stolen) {
			// Closing the queue is the stop wake-up for an idle worker.
			auto blocking = ownQueue->Recv();
			if (blocking) executeOrCancel(*blocking);
		}
	}

	if (InState->DrainOnStop.load(std::memory_order_acquire)) {
		while (true) {
			auto job = ownQueue->TryRecv();
			if (!job) break;
			if (job->Run) job->Run();
		}
	}

	if (InState->WorkersRemaining.fetch_sub(
		1, std::memory_order_acq_rel) == 1) {
		std::lock_guard<std::mutex> lock(InState->FinishedMutex);
		InState->Finished.notify_all();
	}
}

OaThreadPool OaThreadPool::Create(const OaThreadPoolConfig& InConfig) {
	OaThreadPool pool;
	pool.Topology_ = OaCpuTopology::Detect();
	pool.State_ = OaMakeSharedPtr<State>();

	OaI32 numWorkers = InConfig.NumWorkers;
	if (numWorkers <= 0) {
		auto pcores = pool.Topology_.GetPcoreIds();
		numWorkers = static_cast<OaI32>(pcores.Size());
		if (numWorkers <= 0)
			numWorkers = pool.Topology_.NumLogicalCores;
		if (numWorkers <= 0)
			numWorkers = 4;
	}

	// Determine core assignments
	OaVec<OaI32> workerCoreIds;
	if (!InConfig.CoreIds.Empty()) {
		workerCoreIds = InConfig.CoreIds;
	} else if (InConfig.UseTopology) {
		auto pcores = pool.Topology_.GetPcoreIds();
		auto ecores = pool.Topology_.GetEcoreIds();
		workerCoreIds.Reserve(numWorkers);
		for (OaI32 i = 0; i < numWorkers; ++i) {
			if (i < static_cast<OaI32>(pcores.Size()))
				workerCoreIds.PushBack(pcores[i]);
			else if (!ecores.Empty())
				workerCoreIds.PushBack(ecores[i % ecores.Size()]);
			else
				workerCoreIds.PushBack(i % pool.Topology_.NumLogicalCores);
		}
	}

	pool.State_->Running.store(true, std::memory_order_release);
	pool.State_->WorkersRemaining.store(numWorkers, std::memory_order_release);
	pool.State_->Queues.Reserve(numWorkers);
	for (OaI32 i = 0; i < numWorkers; ++i) {
		pool.State_->Queues.PushBack(
			OaMakeSharedPtr<OaChannel<Job>>(kQueueCapacity));
	}

	for (OaI32 i = 0; i < numWorkers; ++i) {
		const OaI32 coreId = i < static_cast<OaI32>(workerCoreIds.Size())
			? workerCoreIds[i] : -1;
		std::thread(
			&OaThreadPool::WorkerLoop,
			pool.State_,
			i,
			coreId,
			InConfig.PinToCores).detach();
	}

	OA_LOG_INFO(OaLogComponent::Core, "ThreadPool: %d workers started", numWorkers);
	return pool;
}

void OaThreadPool::Shutdown() {
	auto state = State_;
	if (!state) return;
	state->DrainOnStop.store(true, std::memory_order_release);
	if (state->Running.exchange(false, std::memory_order_acq_rel)) {
		for (auto& queue : state->Queues) queue->Close();
	}
	std::unique_lock<std::mutex> lock(state->FinishedMutex);
	state->Finished.wait(lock, [&state] {
		return state->WorkersRemaining.load(std::memory_order_acquire) == 0;
	});

	OA_LOG_INFO(OaLogComponent::Core, "ThreadPool: shutdown complete");
}

OaThreadPool::~OaThreadPool() {
	Abandon_();
}

void OaThreadPool::Abandon_() noexcept {
	auto state = OaStdMove(State_);
	if (!state) return;
	state->DrainOnStop.store(false, std::memory_order_release);
	if (state->Running.exchange(false, std::memory_order_acq_rel)) {
		for (auto& queue : state->Queues) queue->Close();
	}
	for (auto& queue : state->Queues) {
		while (auto job = queue->TryRecv()) {
			if (job->Cancel) job->Cancel();
		}
	}
}

OaThreadPool::OaThreadPool(OaThreadPool&& InOther) noexcept
	: State_(OaStdMove(InOther.State_))
	, Topology_(std::move(InOther.Topology_))
{}

OaThreadPool& OaThreadPool::operator=(OaThreadPool&& InOther) noexcept {
	if (this != &InOther) {
		Abandon_();
		State_ = OaStdMove(InOther.State_);
		Topology_ = std::move(InOther.Topology_);
	}
	return *this;
}

void OaThreadPool::Submit(std::function<void()> InJob) {
	SubmitJob_({.Run = OaStdMove(InJob), .Cancel = {}});
}

void OaThreadPool::SubmitJob_(Job InJob) {
	auto state = State_;
	if (!state || !state->Running.load(std::memory_order_acquire)) {
		if (InJob.Cancel) InJob.Cancel();
		return;
	}
	const OaI32 numQ = static_cast<OaI32>(state->Queues.Size());
	const OaI32 idx = state->NextWorker.fetch_add(
		1, std::memory_order_relaxed) % numQ;
	if (state->Queues[idx]->TrySend(InJob)) return;
	for (OaI32 i = 1; i < numQ; ++i) {
		const OaI32 alt = (idx + i) % numQ;
		if (state->Queues[alt]->TrySend(InJob)) return;
	}
	auto cancel = InJob.Cancel;
	if (!state->Queues[idx]->Send(OaStdMove(InJob)) && cancel) cancel();
}

OaI32 OaThreadPool::NumWorkers() const {
	return State_ ? static_cast<OaI32>(State_->Queues.Size()) : 0;
}

bool OaThreadPool::IsRunning() const {
	return State_ && State_->Running.load(std::memory_order_acquire);
}

const OaCpuTopology& OaThreadPool::GetTopology() const {
	return Topology_;
}
