# OA Examples Directory

## Purpose

This directory is for **quick code tests and experiments** without creating a full repository from scratch. Unlike `Tutorial/`, which contains polished, documented learning materials, `Examples/` is your scratch space for:

- Testing new features
- Prototyping ideas
- Quick ML experiments
- Code snippets that don't need full tutorial treatment
- Debugging and validation

## Structure

```
Examples/
├── README.md           # This file
├── CMakeLists.txt      # Build configuration
├── Ml/                 # ML examples (training, inference, etc.)
├── Core/               # Core functionality examples
├── Vision/             # Vision/video examples
├── Scratch/            # Temporary test files (gitignored)
└── ...
```

## Quick Start

### 1. Create a New Example

```cpp
// Examples/Ml/MyQuickTest.cpp
#include "../../Test/OaTest.h"
#include <Oa/Ml.h>  // Includes Training.h

TEST(ExampleMl, MyQuickTest) {
    // Your quick test code here
    auto model = OaMakeSharedPtr<OaLinear>(10, 5);
    auto opt = OaMakeUniquePtr<OaAdamW>(model->AllParameterPtrs(), 0.001F);
    
    OaMlTraining train(model, *opt, OaMlTrainingConfig{
        .TotalSteps = 100,
        .BatchSize = 32
    });
    
    while (train.Step()) {
        // Quick training loop
        train.Next(OaMatrix::Zeros(OaShape1D(1)));
    }
    train.Finish();
}
```

### 2. Build and Run

```bash
# Build
cmake --build Build/Release --target ExampleMl_MyQuickTest

# Run
ctest -R ExampleMl.MyQuickTest
```

## Examples vs Tutorials

| Aspect | Examples/ | Tutorial/ |
|--------|-----------|-----------|
| **Purpose** | Quick tests, prototypes | Learning materials |
| **Documentation** | Minimal or none | Comprehensive |
| **Code Quality** | Experimental | Production-ready |
| **Naming** | Descriptive, flexible | Standardized (Tutorial*) |
| **Lifecycle** | Temporary, can be deleted | Permanent, versioned |
| **Testing** | Optional | Required (CI/CD) |

## When to Use Examples/

✅ **Use Examples/ when:**
- Testing a new feature quickly
- Prototyping an idea
- Debugging a specific issue
- Validating a hypothesis
- Creating throwaway code

❌ **Use Tutorial/ when:**
- Creating learning material
- Demonstrating best practices
- Building reference implementations
- Contributing to documentation
- Code that should be maintained long-term

## Scratch Directory

The `Examples/Scratch/` directory is gitignored for truly temporary files:

```bash
# Create a scratch file
cat > Examples/Scratch/test.cpp << 'EOF'
#include "../../Test/OaTest.h"
TEST(Scratch, QuickTest) {
    // Temporary test code
}
EOF

# Build and run
cmake --build Build/Release --target Scratch_test
ctest -R Scratch.QuickTest

# Delete when done (not tracked by git)
rm Examples/Scratch/test.cpp
```

## Templates

### Minimal ML Example
```cpp
#include "../../Test/OaTest.h"
#include <Oa/Ml.h>               // OaMlTraining (Training.h), modules, OaFnLoss
#include <Oa/Ml/Autograd.h>      // OaGradientTape

TEST(ExampleMl, MinimalTraining) {
    auto model  = OaMakeSharedPtr<OaLinear>(10, 5);
    auto params = model->AllParameterPtrs();
    auto opt    = OaMakeUniquePtr<OaAdamW>(params, 0.001F);

    auto x = OaFnMatrix::RandN(OaShape2D(32, 10));
    auto y = OaFnMatrix::Zeros(OaShape1D(32), OaScalarType::Int32);

    OaMlTraining train(model, *opt, OaMlTrainingConfig{.TotalSteps = 100, .BatchSize = 32});
    while (train.Step()) {
        opt->ZeroGrad();
        OaGradientTape tape;
        auto loss = OaFnLoss::CrossEntropy(model->Forward(x), y);
        tape.Backward(loss);
        train.Next(loss);
    }
    ASSERT_TRUE(train.Finish().IsOk());
}
```

### Minimal Compute Example
```cpp
#include "../../Test/OaTest.h"
#include <Oa/Runtime.h>
#include <Oa/Core.h>

TEST(ExampleCore, MinimalCompute) {
    auto result = OaEngine::Create({.AppName = "ExampleCore"});
    ASSERT_TRUE(result.IsOk());
    auto engine = std::move(*result);
    
    auto a = OaMatrix::Randn(OaShape2D(100, 100));
    auto b = OaMatrix::Randn(OaShape2D(100, 100));
    auto c = OaFnMatrix::MatMul(a, b);
    
    ASSERT_TRUE(engine->GetContext().Execute().IsOk());
    ASSERT_TRUE(engine->GetContext().Sync().IsOk());
    
    EXPECT_EQ(c.Size(0), 100);
    EXPECT_EQ(c.Size(1), 100);
}
```

## Best Practices

1. **Keep it simple** — Examples should be minimal and focused
2. **Use OaMlTraining** — Leverage the simplified API
3. **No boilerplate** — Focus on the code you're testing
4. **Clean up** — Delete examples when done (or move to Scratch/)
5. **Graduate to Tutorial/** — If an example becomes valuable, polish it and move it

## Migration Path

If an example proves valuable:

1. **Polish the code** — Add documentation, error handling
2. **Follow template** — Use `Tutorial/Ml/MlTutorialTemplate.h`
3. **Add to Tutorial/** — Move to appropriate Tutorial/ subdirectory
4. **Update CMakeLists.txt** — Add to Tutorial/CMakeLists.txt
5. **Document** — Add to README or create companion .md file

## See Also

- `Tutorial/` — Polished learning materials
- `Test/` — Unit tests and validation
- `Tutorial/Ml/MlTutorialTemplate.h` — Tutorial template
- [OA developer documentation](https://dev.realm.software/) — training API and guides
- `Examples/Ml/MlTrainingSimple.cpp` — OaMlTraining usage examples
