import("stdfaust.lib");

mix = hslider("mix", 0.48, 0, 1, 0.01);
d = 0.31 * ma.SR;
wet(x) = (x + (x : @(d)) + (x : @(d * 1.013))) / 3;
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
