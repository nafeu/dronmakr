import("stdfaust.lib");

mix = hslider("mix", 0.58, 0, 1, 0.01);
d1 = 107;
d2 = 251;
d3 = 421;
wet(x) = x : @(d1) : @(d2) : @(d3);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
