declare name "Pulse Drift";
declare description "Slowly pulsing filtered square drone.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
lfoRate = hslider("pulse", 0.12, 0.02, 1.5, 0.01);
cutoff = hslider("cutoff", 900, 150, 4000, 1);

envelope = gain * en.adsr(0.35, 0.6, 0.65, 1.5, gate);
trem = 0.55 + 0.45 * os.lf_triangle(lfoRate);
process = os.square(freq) : fi.lowpass(2, cutoff) * envelope * trem <: _, _;
