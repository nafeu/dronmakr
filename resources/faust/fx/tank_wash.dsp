import("stdfaust.lib");

mix = hslider("mix", 0.62, 0, 1, 0.01);
feedback = hslider("feedback", 0.78, 0.4, 0.92, 0.01);
cutoff = hslider("cutoff", 4200, 400, 12000, 1);
tank(x) = x : (+ : @(997)) ~ *(feedback) : @(1201) : @(1597) : @(2111) : @(2707) : fi.lowpass(2, cutoff);
process = tank * mix + _ * (1 - mix);
