import("stdfaust.lib");

room = hslider("room", 0.975, 0.5, 0.995, 0.001);
damp = hslider("damp", 0.28, 0, 1, 0.01);
spread = hslider("spread", 0.78, 0, 1, 0.01);
mix = hslider("mix", 0.64, 0, 1, 0.01);
wet(x) = x : re.mono_freeverb(room, damp, 0.75, spread);
process(l, r) = wet(l) * mix + l * (1 - mix), wet(r) * mix + r * (1 - mix);
