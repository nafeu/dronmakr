import("stdfaust.lib");

room = hslider("room", 0.992, 0.5, 0.998, 0.001);
damp = hslider("damp", 0.12, 0, 1, 0.01);
spread = hslider("spread", 0.95, 0, 1, 0.01);
mix = hslider("mix", 0.78, 0, 1, 0.01);
wet(x) = x : re.mono_freeverb(room, damp, 0.88, spread);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
