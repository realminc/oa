// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — OaRuntime Simple Example
// Demonstrates the new high-level OaRuntime wrapper for zero-boilerplate init
// ═══════════════════════════════════════════════════════════════════════════

#include "../../Test/OaTest.h"
#include <Oa/Oa.h>

// ═══════════════════════════════════════════════════════════════════════════
// Example 1: Simplest possible usage - auto-init with defaults
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, SimplestUsage) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaRuntime Simplest Usage                         ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// Just create OaRuntime - everything else is automatic
	OaRuntime rt;

	printf("Device: %s\n", rt.GetDeviceName().Data());
	printf("Precision: %s\n\n", 
		rt.GetPrecision() == OaPrecision::FP32 ? "FP32" : "BF16");

	// Use default context for operations
	auto a = OaFnMatrix::RandN(OaMatrixShape{64, 128});
	auto b = OaFnMatrix::RandN(OaMatrixShape{64, 128});
	auto c = OaFnMatrix::Add(a, b);
	auto d = OaFnMatrix::Relu(c);

	printf("Created matrices: a[64,128], b[64,128]\n");
	printf("Computed: d = Relu(a + b)\n");
	printf("Result shape: [%lld, %lld]\n\n", 
		static_cast<long long>(d.Size(0)),
		static_cast<long long>(d.Size(1)));

	// Execute and sync happen automatically on scope exit
	EXPECT_EQ(d.Size(0), 64);
	EXPECT_EQ(d.Size(1), 128);
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 2: Using macros for even simpler initialization
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, MacroUsage) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaRuntime Macro Usage                            ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// Single line initialization with BF16 precision
	OA_INIT_RUNTIME_BF16();

	printf("Initialized with BF16 precision using macro\n\n");

	auto x = OaFnMatrix::Ones(OaMatrixShape{32, 32});
	auto y = OaFnMatrix::Mul(x, 2.0F);

	printf("Created x = Ones[32,32]\n");
	printf("Computed y = x * 2.0\n");
	printf("y[0,0] = %.2f (expected 2.0)\n\n", y.At(0));

	EXPECT_NEAR(y.At(0), 2.0F, 1e-3F);
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 3: Manual execution control
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, ManualControl) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaRuntime Manual Control                         ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// Disable auto-execute for manual control
	OaRuntime rt(OaRuntimeConfig{.AutoExecute = false});

	printf("Manual execution mode enabled\n\n");

	// Record operations
	auto a = OaFnMatrix::RandN(OaMatrixShape{100, 100});
	auto b = OaFnMatrix::RandN(OaMatrixShape{100, 100});
	auto c = OaFnMatrix::MatMulNt(a, b);

	printf("Recorded: c = MatMul(a, b)\n");
	printf("Operations queued: %u\n\n", rt.GetContext().NodeCount());

	// Manually execute
	auto status = rt.Execute();
	ASSERT_TRUE(status.IsOk()) << "Execute failed: " << status.ToString();

	printf("Executed %u operations\n", rt.GetContext().NodeCount());

	// Manually sync
	status = rt.Sync();
	ASSERT_TRUE(status.IsOk()) << "Sync failed: " << status.ToString();

	printf("Synchronized with GPU\n");
	printf("Result shape: [%lld, %lld]\n\n",
		static_cast<long long>(c.Size(0)),
		static_cast<long long>(c.Size(1)));

	EXPECT_EQ(c.Size(0), 100);
	EXPECT_EQ(c.Size(1), 100);
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 4: Configuration options
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, ConfigurationOptions) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaRuntime Configuration                          ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// Custom configuration
	OaRuntimeConfig cfg;
	cfg.Precision = OaPrecision::FP32;
	cfg.AppName = "TutorialApp";
	cfg.EnableValidation = false;
	cfg.DeviceIndex = -1;  // Auto-select

	OaRuntime rt(cfg);

	printf("Configuration:\n");
	printf("  App Name: %s\n", cfg.AppName.c_str());
	printf("  Precision: FP32\n");
	printf("  Device: %s\n", rt.GetDeviceName().Data());
	printf("  Validation: %s\n\n", cfg.EnableValidation ? "Enabled" : "Disabled");

	// Verify configuration
	EXPECT_EQ(rt.GetPrecision(), OaPrecision::FP32);
	EXPECT_TRUE(rt.IsValid());
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 5: RAII scope for isolated compute blocks
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, ScopeUsage) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaRuntime Scope Usage                            ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	OaMatrix result;

	{
		// Scoped runtime - auto cleanup
		OaRuntime::Scope scope;
		
		printf("Inside scope - runtime initialized\n");

		auto x = OaFnMatrix::RandN(OaMatrixShape{50, 50});
		auto y = OaFnMatrix::Sum(x);
		result = y;

		printf("Computed sum of 50x50 random matrix\n");
		// Auto Execute() + Sync() on scope exit
	}

	printf("Outside scope - runtime cleaned up\n");
	printf("Result preserved: %.4f\n\n", result.At(0));

	EXPECT_TRUE(std::isfinite(result.At(0)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 6: Comparison with old style (for reference)
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialCoreRuntime, ComparisonWithOldStyle) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Old vs New Style Comparison                      ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	printf("OLD STYLE (verbose):\n");
	printf("  OaEngineConfig cfg;\n");
	printf("  cfg.Precision = OaPrecision::BF16;\n");
	printf("  auto result = OaComputeEngine::Create(cfg);\n");
	printf("  OaComputeEngine* engine = result.GetValue().Release();\n");
	printf("  auto context = OaContext::Create(engine);\n");
	printf("  // ... work ...\n");
	printf("  context->Execute();\n");
	printf("  context->Sync();\n");
	printf("  // ... cleanup ...\n\n");

	printf("NEW STYLE (simple):\n");
	printf("  OaRuntime rt;\n");
	printf("  // ... work ...\n");
	printf("  // Auto Execute() + Sync() + cleanup\n\n");

	// Demonstrate the new style
	OaRuntime rt;
	auto x = OaFnMatrix::Ones(OaMatrixShape{10, 10});
	auto y = OaFnMatrix::Sum(x);

	printf("Result: sum of 10x10 ones = %.1f (expected 100.0)\n\n", y.At(0));
	EXPECT_NEAR(y.At(0), 100.0F, 1e-3F);
}

