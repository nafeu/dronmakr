import("stdfaust.lib");

room = hslider("room", 0.988, 0.5, 0.998, 0.001);
damp = hslider("damp", 0.58, 0, 1, 0.01);
spread = hslider("spread", 0.65, 0, 1, 0.01);
cutoff = hslider("cutoff", 2800, 400, 8000, 1);
mix = hslider("mix", 0.7, 0, 1, 0.01);
wet(x) = re.mono_freeverb(room, damp, 0.8, spread) : fi.lowpass(2, cutoff);
process = wet * mix + _ * (1 - mix);
