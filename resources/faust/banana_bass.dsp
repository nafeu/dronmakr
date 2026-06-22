declare name "Banana Bass";
declare description "Rubbery resonant bass squelch.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 680, 120, 4000, 1);
res = hslider("res", 0.85, 0.3, 0.98, 0.01);
envelope = gain * en.adsr(0.01, 0.1, 0.62, 0.22, gate);
process = os.sawtooth(freq) : fi.resonlp(2, cut, res) * envelope <: _, _;
