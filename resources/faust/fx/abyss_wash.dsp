import("stdfaust.lib");

size = hslider("size", 10, 2, 20, 0.1);
damp = hslider("damp", 0.45, 0, 1, 0.01);
mix = hslider("mix", 0.75, 0, 1, 0.01);
wet = re.greyhole(size, damp, 0.98, 0.48, 0.96, 3200, 1);
mono = wet :> _/2;
process = mono * mix + _ * (1 - mix);
