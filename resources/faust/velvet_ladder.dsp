declare name "Velvet Ladder";
declare description "Smooth resonant low-pass on a soft saw.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 1500, 200, 7000, 1);
res = hslider("res", 0.45, 0.05, 0.9, 0.01);
envelope = gain * en.adsr(0.1, 0.35, 0.78, 0.55, gate);
process = os.sawtooth(freq) : fi.resonlp(2, cut, res) * envelope * 0.75 <: _, _;
