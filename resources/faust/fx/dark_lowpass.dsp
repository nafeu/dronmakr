import("stdfaust.lib");

cutoff = hslider("cutoff", 1400, 80, 8000, 1);
mix = hslider("mix", 0.88, 0, 1, 0.01);
wet(x) = fi.lowpass(3, cutoff);
process = wet * mix + _ * (1 - mix);
