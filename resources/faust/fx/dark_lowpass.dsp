import("stdfaust.lib");

cutoff = hslider("cutoff", 1400, 80, 8000, 1);
mix = hslider("mix", 0.88, 0, 1, 0.01);
wet(x) = x : fi.lowpass(3, cutoff);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
