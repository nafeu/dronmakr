import("stdfaust.lib");

mix = hslider("mix", 0.48, 0, 1, 0.01);
d = 0.31 * ma.SR;
t1 = de.delay(d);
t2 = de.delay(d * 1.013);
wet = (_ + t1 + t2) / 3;
process = wet * mix + _ * (1 - mix);
