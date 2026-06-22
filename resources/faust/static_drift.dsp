declare name "Static Drift";
declare description "Crackly noise with a wandering band-pass.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 1400, 300, 6000, 1);
drift = hslider("drift", 0.4, 0, 1, 0.01);

envelope = gain * en.adsr(0.15, 0.45, 0.7, 0.9, gate);
wander = cutoff * (1 + drift * os.lf_triangle(0.18));
process = no.noise : fi.resonbp(2, wander, 2.5) * envelope * 0.5 <: _, _;
