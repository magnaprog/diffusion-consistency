# Figure fonts and reproduction commands

If a figure reports a font error, check which DejaVu Serif file Matplotlib finds in the Python environment where you run the plot.

```bash
python -c "from matplotlib import font_manager; print(font_manager.findfont('DejaVu Serif', fallback_to_default=False))"
```

The path should end in `.ttf`, the extension the renderers check for a TrueType font. Figure 3 also uses Nimbus Roman in its table. The reproduction guide gives the TeX Live requirements and the command for each figure. [Draw the figures](reproduction.md#draw-the-figures)

The plotting scripts set their own method colors. For example, Figures 7–9 use pink for O-S and orange for G5. The color cycle in `prigm.mplstyle` supplies defaults for plots that do not set a color explicitly. [Figure code](../figures/)

## Paths in the original study code

The source comments refer to the manuscript and study plan by their paths in the development repository. In this checkout, `src/` is the directory containing `prigm/`. If you call `frozen_source_snapshot` directly, pass that directory as its root. The stored module names start with `prigm/`. [Checkpoint utilities](../src/prigm/checkpoint.py) · [Public reproduction guide](reproduction.md)
