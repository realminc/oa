// Oa/Plot — matplotlib-style plotting on top of OaContext sinks.
//
// UnifiedExecutionArchitecture.md §3.5 / OaUiFinalGlueBridge.md §5.4 (Step 3e). The module is
// a thin layout + replay layer over the unified OaContext recorder:
//
//   OaPlot::Figure fig({.Rows=5, .Cols=5, .Width=800, .Height=800});
//   for (int i = 0; i < 25; ++i) {
//     auto& ax = fig.Ax(i / 5, i % 5);
//     ax.Imshow(tiles[i]);
//     ax.Title(classNames[pred[i]],
//              correct ? kSuccess : kError);
//   }
//   (void)fig.Show();            // window sink
//   (void)fig.SaveFig("g.png");  // batch sink
//
// Phase-1 surface: Imshow + text labels work end-to-end. Plot/Bar/Scatter
// land in Show via OaUi::PlotLine; they are skipped in SaveFig until the
// Phase-2 MSDF text rasterizer + SDF widgets land.

#pragma once

#include <Oa/Ui/Plot/Figure.h>
#include <Oa/Ui/Plot/Axes.h>
