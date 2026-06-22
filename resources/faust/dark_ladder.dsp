declare name "Dark Ladder";
declare description "Deep resonant ladder filter on a square.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 520, 80, 3000, 1);
res = hslider("res", 0.78, 0.2, 0.95, 0.01);
envelope = gain * en.adsr(0.04, 0.18, 0.72, 0.35, gate);
process = os.square(freq) : fi.resonlp(3, cut, res) * envelope <: _, _;
