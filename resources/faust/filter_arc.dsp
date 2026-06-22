declare name "Filter Arc";
declare description "Sweeping resonant triangle with bite.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 1800, 300, 8000, 1);
res = hslider("res", 0.55, 0.1, 0.92, 0.01);
envelope = gain * en.adsr(0.04, 0.16, 0.76, 0.32, gate);
process = os.triangle(freq) : fi.resonlp(2, cut, res) * envelope <: _, _;
