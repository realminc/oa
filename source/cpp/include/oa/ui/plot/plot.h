// oa::plot — matplotlib-style plotting with explicit engine-owned sinks.
//
// architecture/oaArchitecture.md §10. The module is
// a thin layout + replay layer over the engine's private recorder:
//
//   oa::plot::Figure fig({.rows=5, .cols=5, .width=800, .height=800});
//   for (int i = 0; i < 25; ++i) {
//     auto& ax = fig.ax(i / 5, i % 5);
//     ax.imshow(tiles[i]);
//     ax.title(classNames[pred[i]],
//              correct ? kSuccess : kError);
//   }
//   (void)fig.show();            // window sink
//   (void)fig.saveTo("g.png");  // batch sink
//
// Compact surface: raster bases plus ordered line, scatter, bar, and histogram
// artists work in interactive and headless sinks. Explicit-X curves, legends,
// labels, themes, and generated text share the same GPU-composed replay.

#pragma once

#include <oa/ui/plot/figure.h>
#include <oa/ui/plot/axes.h>
