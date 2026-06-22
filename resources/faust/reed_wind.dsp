declare name "Reed Wind";
declare description "Nasal reed tone from pulse and triangle mix.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 1800, 300, 7000, 1);

envelope = gain * en.adsr(0.04, 0.15, 0.82, 0.35, gate);
voice = os.triangle(freq) + 0.6 * os.square(freq);
process = voice : fi.lowpass(2, cutoff) * envelope * 0.7 <: _, _;
