declare name "Glass Pad";
declare description "Bright, airy pad with a slow bloom.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 5200, 800, 12000, 1);

envelope = gain * en.adsr(0.35, 0.45, 0.75, 1.2, gate);
voice = os.triangle(freq) + 0.35 * os.osc(freq * 2.01);
process = voice : fi.lowpass(2, cutoff) * envelope * 0.7 <: _, _;
