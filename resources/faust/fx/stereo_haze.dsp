import("stdfaust.lib");

rate = hslider("rate", 0.18, 0.02, 1, 0.01);
depth = hslider("depth", 0.003, 0, 0.012, 0.0001);
mix = hslider("mix", 0.52, 0, 1, 0.01);
lfo = os.osc(rate) * depth;
wet(x) = x + de.fdelay(512, 0.011 + lfo, x) * 0.42;
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
