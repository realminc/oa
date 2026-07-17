// Oa/Plot — matplotlib-style plotting on top of OaContext sinks.
//
// Architecture/OaArchitecture.md §10. The module is
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
// Compact surface: Imshow, line curves and heatmaps work in interactive and
// headless sinks. Text labels are interactive; headless glyph composition is
// the remaining presentation upgrade.

#pragma once

#include <Oa/Ui/Plot/Figure.h>
#include <Oa/Ui/Plot/Axes.h>
