#include <oa/core/thread.h>
#include <oa/core/log.h>

#include <condition_variable>
#include <mutex>

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
	std::atomic<oa::I32> nextWorker{0};
	std::atomic<oa::I32> workersRemaining{0};
	std::atomic<bool> running{false};
	std::atomic<bool> drainOnStop{false};
	std::mutex finishedMutex;
	std::condition_variable finished;
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
		if (inState->running.load(std::memory_order_acquire)
			|| inState->drainOnStop.load(std::memory_order_acquire)) {
			if (inJob.run) inJob.run();
		} else if (inJob.cancel) {
			inJob.cancel();
		}
	};

	while (inState->running.load(std::memory_order_acquire)) {
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

	if (inState->drainOnStop.load(std::memory_order_acquire)) {
		while (true) {
			auto job = ownQueue->tryRecv();
			if (!job) break;
			if (job->run) job->run();
		}
	}

	if (inState->workersRemaining.fetch_sub(
		1, std::memory_order_acq_rel) == 1) {
		std::lock_guard<std::mutex> lock(inState->finishedMutex);
		inState->finished.notify_all();
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

	pool.state_->running.store(true, std::memory_order_release);
	pool.state_->workersRemaining.store(numWorkers, std::memory_order_release);
	pool.state_->queues.reserve(numWorkers);
	for (oa::I32 i = 0; i < numWorkers; ++i) {
		pool.state_->queues.pushBack(
			oa::makeShared<oa::Channel<Job>>(kQueueCapacity));
	}

	for (oa::I32 i = 0; i < numWorkers; ++i) {
		const oa::I32 coreId = i < static_cast<oa::I32>(workerCoreIds.size())
			? workerCoreIds[i] : -1;
		std::thread(
			&oa::ThreadPool::workerLoop,
			pool.state_,
			i,
			coreId,
			inConfig.pinToCores).detach();
	}

	OaLogInfo(oa::LogComponent::Core, "ThreadPool: %d workers started", numWorkers);
	return pool;
}

void oa::ThreadPool::shutdown() {
	auto state = state_;
	if (!state) return;
	state->drainOnStop.store(true, std::memory_order_release);
	if (state->running.exchange(false, std::memory_order_acq_rel)) {
		for (auto& queue : state->queues) queue->close();
	}
	std::unique_lock<std::mutex> lock(state->finishedMutex);
	state->finished.wait(lock, [&state] {
		return state->workersRemaining.load(std::memory_order_acquire) == 0;
	});

	OaLogInfo(oa::LogComponent::Core, "ThreadPool: shutdown complete");
}

oa::ThreadPool::~ThreadPool() {
	abandon_();
}

void oa::ThreadPool::abandon_() noexcept {
	auto state = oa::move(state_);
	if (!state) return;
	state->drainOnStop.store(false, std::memory_order_release);
	if (state->running.exchange(false, std::memory_order_acq_rel)) {
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
	, topology_(std::move(inOther.topology_))
{}

oa::ThreadPool& oa::ThreadPool::operator=(oa::ThreadPool&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		state_ = oa::move(inOther.state_);
		topology_ = std::move(inOther.topology_);
	}
	return *this;
}

void oa::ThreadPool::submit(std::function<void()> inJob) {
	submitJob_({.run = oa::move(inJob), .cancel = {}});
}

void oa::ThreadPool::submitJob_(Job inJob) {
	auto state = state_;
	if (!state || !state->running.load(std::memory_order_acquire)) {
		if (inJob.cancel) inJob.cancel();
		return;
	}
	const oa::I32 numQ = static_cast<oa::I32>(state->queues.size());
	const oa::I32 idx = state->nextWorker.fetch_add(
		1, std::memory_order_relaxed) % numQ;
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
	return state_ && state_->running.load(std::memory_order_acquire);
}

const oa::CpuTopology& oa::ThreadPool::getTopology() const {
	return topology_;
}
