#include <oa/core/thread.h>
#include <oa/core/log.h>

#ifdef OA_PLATFORM_LINUX
#include <pthread.h>
#include <sched.h>
#endif

static bool pinThreadToCore([[maybe_unused]] oa::I32 inCoreId) {
#ifdef OA_PLATFORM_LINUX
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(inCoreId, &cpuset);
	return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#else
	return false;
#endif
}

struct oa::ThreadPool::State {
	oa::Vec<oa::SharedPtr<oa::Channel<Job>>> queues;
	oa::Atomic<oa::I32> nextWorker{0};
	oa::Atomic<oa::I32> workersRemaining{0};
	oa::Atomic<bool> running{false};
	oa::Atomic<bool> drainOnStop{false};
	oa::Mutex finishedMutex;
	oa::Condition finished;
};

void oa::ThreadPool::workerLoop(
	oa::SharedPtr<State> inState,
	oa::I32 inWorkerId,
	oa::I32 inCoreId,
	oa::Bool inPinToCore)
{
	if (inPinToCore && inCoreId >= 0) {
		pinThreadToCore(inCoreId);
	}

	const oa::I32 numQueues = static_cast<oa::I32>(inState->queues.size());
	auto& ownQueue = inState->queues[inWorkerId];
	auto executeOrCancel = [&inState](Job& inJob) {
		if (inState->running.load(oa::MemoryOrder::Acquire)
			|| inState->drainOnStop.load(oa::MemoryOrder::Acquire)) {
			if (inJob.run) inJob.run();
		} else if (inJob.cancel) {
			inJob.cancel();
		}
	};

	while (inState->running.load(oa::MemoryOrder::Acquire)) {
		// Try own queue first (non-blocking to allow stealing)
		auto job = ownQueue->tryRecv();
		if (job) {
			executeOrCancel(*job);
			continue;
		}

		// Work stealing: try sibling queues
		bool stolen = false;
		for (oa::I32 i = 1; i < numQueues; ++i) {
			oa::I32 target = (inWorkerId + i) % numQueues;
			auto stealJob = inState->queues[target]->tryRecv();
			if (stealJob) {
				executeOrCancel(*stealJob);
				stolen = true;
				break;
			}
		}

		if (!stolen) {
			// Closing the queue is the stop wake-up for an idle worker.
			auto blocking = ownQueue->recv();
			if (blocking) executeOrCancel(*blocking);
		}
	}

	if (inState->drainOnStop.load(oa::MemoryOrder::Acquire)) {
		while (true) {
			auto job = ownQueue->tryRecv();
			if (!job) break;
			if (job->run) job->run();
		}
	}

	if (inState->workersRemaining.fetchSub(
		1, oa::MemoryOrder::AcquireRelease) == 1) {
		oa::ScopedLock<oa::Mutex> lock(inState->finishedMutex);
		inState->finished.notifyAll();
	}
}

oa::ThreadPool oa::ThreadPool::create(const oa::ThreadPoolConfig& inConfig) {
	oa::ThreadPool pool;
	pool.topology_ = oa::CpuTopology::detect();
	pool.state_ = oa::makeShared<State>();

	oa::I32 numWorkers = inConfig.numWorkers;
	if (numWorkers <= 0) {
		auto pcores = pool.topology_.getPcoreIds();
		numWorkers = static_cast<oa::I32>(pcores.size());
		if (numWorkers <= 0)
			numWorkers = pool.topology_.numLogicalCores;
		if (numWorkers <= 0)
			numWorkers = 4;
	}

	// Determine core assignments
	oa::Vec<oa::I32> workerCoreIds;
	if (!inConfig.coreIds.empty()) {
		workerCoreIds = inConfig.coreIds;
	} else if (inConfig.useTopology) {
		auto pcores = pool.topology_.getPcoreIds();
		auto ecores = pool.topology_.getEcoreIds();
		workerCoreIds.reserve(numWorkers);
		for (oa::I32 i = 0; i < numWorkers; ++i) {
			if (i < static_cast<oa::I32>(pcores.size()))
				workerCoreIds.pushBack(pcores[i]);
			else if (!ecores.empty())
				workerCoreIds.pushBack(ecores[i % ecores.size()]);
			else
				workerCoreIds.pushBack(i % pool.topology_.numLogicalCores);
		}
	}

	pool.state_->running.store(true, oa::MemoryOrder::Release);
	pool.state_->workersRemaining.store(numWorkers, oa::MemoryOrder::Release);
	pool.state_->queues.reserve(numWorkers);
	for (oa::I32 i = 0; i < numWorkers; ++i) {
		pool.state_->queues.pushBack(
			oa::makeShared<oa::Channel<Job>>(kQueueCapacity));
	}

	for (oa::I32 i = 0; i < numWorkers; ++i) {
		const oa::I32 coreId = i < static_cast<oa::I32>(workerCoreIds.size())
			? workerCoreIds[i] : -1;
		auto worker = oa::Thread::create([
			state = pool.state_, i, coreId, pinToCores = inConfig.pinToCores
		] {
			oa::ThreadPool::workerLoop(state, i, coreId, pinToCores);
		});
		if (worker.isError()) {
			OaLogError(oa::LogComponent::Core,
				"ThreadPool worker %d failed to start: %s", i,
				worker.getStatus().toString().cStr());
		}
		OA_REQUIRE(worker.isOk());
		oa::Thread thread = oa::move(*worker);
		const oa::Status detachStatus = thread.detach();
		OA_REQUIRE(detachStatus.isOk());
	}

	OaLogInfo(oa::LogComponent::Core, "ThreadPool: %d workers started", numWorkers);
	return pool;
}

void oa::ThreadPool::shutdown() {
	auto state = state_;
	if (!state) return;
	state->drainOnStop.store(true, oa::MemoryOrder::Release);
	if (state->running.exchange(false, oa::MemoryOrder::AcquireRelease)) {
		for (auto& queue : state->queues) queue->close();
	}
	oa::UniqueLock<oa::Mutex> lock(state->finishedMutex);
	state->finished.wait(lock, [&state] {
		return state->workersRemaining.load(oa::MemoryOrder::Acquire) == 0;
	});

	OaLogInfo(oa::LogComponent::Core, "ThreadPool: shutdown complete");
}

oa::ThreadPool::~ThreadPool() {
	abandon_();
}

void oa::ThreadPool::abandon_() noexcept {
	auto state = oa::move(state_);
	if (!state) return;
	state->drainOnStop.store(false, oa::MemoryOrder::Release);
	if (state->running.exchange(false, oa::MemoryOrder::AcquireRelease)) {
		for (auto& queue : state->queues) queue->close();
	}
	for (auto& queue : state->queues) {
		while (auto job = queue->tryRecv()) {
			if (job->cancel) job->cancel();
		}
	}
}

oa::ThreadPool::ThreadPool(oa::ThreadPool&& inOther) noexcept
	: state_(oa::move(inOther.state_))
	, topology_(oa::move(inOther.topology_))
{}

oa::ThreadPool& oa::ThreadPool::operator=(oa::ThreadPool&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		state_ = oa::move(inOther.state_);
		topology_ = oa::move(inOther.topology_);
	}
	return *this;
}

void oa::ThreadPool::submit(oa::Fn<void()> inJob) {
	submitJob_({.run = oa::move(inJob), .cancel = {}});
}

void oa::ThreadPool::submitJob_(Job inJob) {
	auto state = state_;
	if (!state || !state->running.load(oa::MemoryOrder::Acquire)) {
		if (inJob.cancel) inJob.cancel();
		return;
	}
	const oa::I32 numQ = static_cast<oa::I32>(state->queues.size());
	const oa::I32 idx = state->nextWorker.fetchAdd(
		1, oa::MemoryOrder::Relaxed) % numQ;
	if (state->queues[idx]->trySend(inJob)) return;
	for (oa::I32 i = 1; i < numQ; ++i) {
		const oa::I32 alt = (idx + i) % numQ;
		if (state->queues[alt]->trySend(inJob)) return;
	}
	auto cancel = inJob.cancel;
	if (!state->queues[idx]->send(oa::move(inJob)) && cancel) cancel();
}

oa::I32 oa::ThreadPool::numWorkers() const {
	return state_ ? static_cast<oa::I32>(state_->queues.size()) : 0;
}

bool oa::ThreadPool::isRunning() const {
	return state_ && state_->running.load(oa::MemoryOrder::Acquire);
}

const oa::CpuTopology& oa::ThreadPool::getTopology() const {
	return topology_;
}
