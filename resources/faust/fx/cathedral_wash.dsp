import("stdfaust.lib");

room = hslider("room", 0.985, 0.5, 0.995, 0.001);
damp = hslider("damp", 0.18, 0, 1, 0.01);
spread = hslider("spread", 0.92, 0, 1, 0.01);
mix = hslider("mix", 0.72, 0, 1, 0.01);
wet(x) = x : re.mono_freeverb(room, damp, 0.82, spread);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
