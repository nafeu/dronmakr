declare name "Formant Box";
declare description "Formant-like band-pass vocal filter.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 1100, 300, 2800, 1);
envelope = gain * en.adsr(0.05, 0.2, 0.78, 0.4, gate);
process = os.triangle(freq) : fi.resonbp(2, cut, 5) * envelope * 0.7 <: _, _;
