// AUTO-GENERATED umbrella — includes all autograd sub-domain headers.
//
// Source of truth: individual Autograd* .gen.h files under subdirectories.
// Regenerate via: python3 Tools/FnAutogen/oafnautogen.py --live

#pragma once

// Public tape contract (hand-written)
#include <Oa/Ml/Autograd.h>

// Matrix domain — generated sub-categories
#include "Matrix/AutogradElemwise.gen.h"
#include "Matrix/AutogradActivation.gen.h"

// Matrix / activation / conv / norm / pool grads (hand-written)
#include "Matrix/AutogradMatrix.h"

// Loss grads (hand-written)
#include "Loss/AutogradLoss.h"
