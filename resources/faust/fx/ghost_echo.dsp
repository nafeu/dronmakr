import("stdfaust.lib");

delay_ms = hslider("delay_ms", 480, 40, 1400, 1);
feedback = hslider("feedback", 0.44, 0, 0.92, 0.01);
mix = hslider("mix", 0.42, 0, 1, 0.01);
d = delay_ms * ma.SR / 1000;
wet(x) = x + (x : @(d) : *(feedback)) : fi.lowpass(2, 3200);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
