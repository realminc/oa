#include <Oa/Core/Thread.h>
#include <Oa/Core/Log.h>

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

static void WorkerLoop(
	OaI32 InWorkerId,
	OaSharedPtr<OaChannel<std::function<void()>>>& InOwnQueue,
	OaVec<OaSharedPtr<OaChannel<std::function<void()>>>>& InAllQueues,
	std::atomic<bool>& InRunning,
	OaI32 InCoreId,
	OaBool InPinToCore
) {
	if (InPinToCore && InCoreId >= 0) {
		PinThreadToCore(InCoreId);
	}

	OaI32 numQueues = static_cast<OaI32>(InAllQueues.Size());

	while (InRunning.load(std::memory_order_relaxed)) {
		// Try own queue first (non-blocking to allow stealing)
		auto job = InOwnQueue->TryRecv();
		if (job) {
			(*job)();
			continue;
		}

		// Work stealing: try sibling queues
		bool stolen = false;
		for (OaI32 i = 1; i < numQueues; ++i) {
			OaI32 target = (InWorkerId + i) % numQueues;
			auto stealJob = InAllQueues[target]->TryRecv();
			if (stealJob) {
				(*stealJob)();
				stolen = true;
				break;
			}
		}

		if (!stolen) {
			// No work anywhere — block on own queue briefly
			auto blocking = InOwnQueue->Recv();
			if (blocking) (*blocking)();
		}
	}

	// Drain remaining jobs
	while (true) {
		auto job = InOwnQueue->TryRecv();
		if (!job) break;
		(*job)();
	}
}

OaThreadPool OaThreadPool::Create(const OaThreadPoolConfig& InConfig) {
	OaThreadPool pool;
	pool.Topology_ = OaCpuTopology::Detect();

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
	if (!InConfig.CoreIds.Empty()) {
		pool.WorkerCoreIds_ = InConfig.CoreIds;
	} else if (InConfig.UseTopology) {
		auto pcores = pool.Topology_.GetPcoreIds();
		auto ecores = pool.Topology_.GetEcoreIds();
		pool.WorkerCoreIds_.Reserve(numWorkers);
		for (OaI32 i = 0; i < numWorkers; ++i) {
			if (i < static_cast<OaI32>(pcores.Size()))
				pool.WorkerCoreIds_.PushBack(pcores[i]);
			else if (!ecores.Empty())
				pool.WorkerCoreIds_.PushBack(ecores[i % ecores.Size()]);
			else
				pool.WorkerCoreIds_.PushBack(i % pool.Topology_.NumLogicalCores);
		}
	}

	pool.Running_.store(true, std::memory_order_release);
	pool.Queues_.Reserve(numWorkers);
	for (OaI32 i = 0; i < numWorkers; ++i) {
		pool.Queues_.PushBack(OaMakeSharedPtr<OaChannel<Job>>(kQueueCapacity));
	}

	pool.Workers_.Reserve(numWorkers);
	for (OaI32 i = 0; i < numWorkers; ++i) {
		OaI32 coreId = (i < static_cast<OaI32>(pool.WorkerCoreIds_.Size()))
			? pool.WorkerCoreIds_[i] : -1;
		pool.Workers_.EmplaceBack(WorkerLoop,
			i,
			std::ref(pool.Queues_[i]),
			std::ref(pool.Queues_),
			std::ref(pool.Running_),
			coreId,
			InConfig.PinToCores);
	}

	OA_LOG_INFO(OaLogComponent::Core, "ThreadPool: %d workers started", numWorkers);
	return pool;
}

void OaThreadPool::Shutdown() {
	if (!Running_.exchange(false)) return;

	for (auto& q : Queues_) q->Close();
	for (auto& w : Workers_) {
		if (w.joinable()) w.join();
	}

	OA_LOG_INFO(OaLogComponent::Core, "ThreadPool: shutdown complete");
}

OaThreadPool::~OaThreadPool() {
	if (Running_.load()) Shutdown();
}

OaThreadPool::OaThreadPool(OaThreadPool&& InOther) noexcept
	: Workers_(std::move(InOther.Workers_))
	, Queues_(std::move(InOther.Queues_))
	, NextWorker_(InOther.NextWorker_.load())
	, Running_(InOther.Running_.load())
	, Topology_(std::move(InOther.Topology_))
	, WorkerCoreIds_(std::move(InOther.WorkerCoreIds_))
{
	InOther.Running_.store(false);
}

OaThreadPool& OaThreadPool::operator=(OaThreadPool&& InOther) noexcept {
	if (this != &InOther) {
		if (Running_.load()) Shutdown();
		Workers_ = std::move(InOther.Workers_);
		Queues_ = std::move(InOther.Queues_);
		NextWorker_.store(InOther.NextWorker_.load());
		Running_.store(InOther.Running_.load());
		Topology_ = std::move(InOther.Topology_);
		WorkerCoreIds_ = std::move(InOther.WorkerCoreIds_);
		InOther.Running_.store(false);
	}
	return *this;
}

void OaThreadPool::Submit(std::function<void()> InJob) {
	if (!Running_.load(std::memory_order_relaxed)) return;
	OaI32 numQ = static_cast<OaI32>(Queues_.Size());
	OaI32 idx = NextWorker_.fetch_add(1, std::memory_order_relaxed) % numQ;
	if (Queues_[idx]->TrySend(InJob)) return;
	for (OaI32 i = 1; i < numQ; ++i) {
		OaI32 alt = (idx + i) % numQ;
		if (Queues_[alt]->TrySend(InJob)) return;
	}
	Queues_[idx]->Send(std::move(InJob));
}

OaI32 OaThreadPool::NumWorkers() const {
	return static_cast<OaI32>(Workers_.Size());
}

bool OaThreadPool::IsRunning() const {
	return Running_.load(std::memory_order_relaxed);
}

const OaCpuTopology& OaThreadPool::GetTopology() const {
	return Topology_;
}
