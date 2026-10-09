//! `oa::plot` — matplotlib-style retained figure and axes.
//!
//! The module is a thin layout + replay layer over the engine's GPU
//! compositor.  Three terminal sinks are provided:
//!
//! ```rust,ignore
//! let mut fig = plot::Figure::new(plot::FigureConfig { rows: 2, cols: 2, ..Default::default() });
//! fig.ax(0, 0).plot(&loss_data, Default::default());
//! fig.ax(0, 1).imshow(texture);
//! fig.save_to(&engine, "out.png")?;   // headless file sink
//! let img = fig.render(&engine)?;     // → oa::Image
//! fig.show(&engine)?;                 // interactive window
//! ```
//!
//! Compact surface: raster bases plus ordered line, scatter, bar, and
//! histogram artists share the same GPU-composed replay.
//!
//! Donor reference: `oa/ui/plot/plot.h`

pub mod axes;
pub mod figure;

pub use axes::{Axes, BarStyle, HeatmapStyle, LineStyle, ScatterStyle};
pub use figure::{Figure, FigureConfig, Theme};
