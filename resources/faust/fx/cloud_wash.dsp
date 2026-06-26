import("stdfaust.lib");

size = hslider("size", 6.5, 1, 15, 0.05);
damp = hslider("damp", 0.2, 0, 1, 0.01);
mix = hslider("mix", 0.72, 0, 1, 0.01);
wet = re.greyhole(size, damp, 0.96, 0.62, 0.94, 7500, 1);
mono = wet :> _/2;
process = mono * mix + _ * (1 - mix);
