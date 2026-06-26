import("stdfaust.lib");

rate = hslider("rate", 0.28, 0.05, 1.5, 0.01);
depth = hslider("depth", 0.004, 0, 0.015, 0.0001);
base_ms = hslider("delay_ms", 18, 5, 45, 0.5);
mix = hslider("mix", 0.62, 0, 1, 0.01);
mod = os.osc(rate) * depth + base_ms * ma.SR / 1000;
wet(x) = de.fdelay(2048, mod, x);
process = wet * mix + _ * (1 - mix);
