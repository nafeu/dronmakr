import("stdfaust.lib");

freq = hslider("freq", 280, 80, 1200, 1);
q = hslider("q", 0.85, 0.2, 3, 0.01);
mix = hslider("mix", 0.68, 0, 1, 0.01);
wet(x) = x : fi.resonbp(2, freq, q);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
