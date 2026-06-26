import("stdfaust.lib");

mix = hslider("mix", 0.45, 0, 1, 0.01);
c1 = 97;
c2 = 163;
c3 = 241;
wet(x) = (x + x : @(c1) + x : @(c2) + x : @(c3)) / 4;
process = wet * mix + _ * (1 - mix);
