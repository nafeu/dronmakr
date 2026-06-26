import("stdfaust.lib");

drive = hslider("drive", 1.8, 0.5, 6, 0.01);
mix = hslider("mix", 0.82, 0, 1, 0.01);
wet(x) = x * drive / (1 + abs(x * drive));
process = wet * mix + _ * (1 - mix);
