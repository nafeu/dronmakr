declare name "Radio Static";
declare description "Tuned static bursts through a band-pass.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 2200, 400, 8000, 1);
envelope = gain * en.adsr(0.05, 0.25, 0.55, 0.6, gate);
process = no.noise : fi.resonbp(2, cut, 2.5) * envelope * 0.45 <: _, _;
