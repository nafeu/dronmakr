declare name "Notch Scrape";
declare description "Band-pass saw scrape with a vocal edge.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 1200, 200, 6000, 1);
q = hslider("q", 3.5, 0.5, 12, 0.01);
envelope = gain * en.adsr(0.03, 0.15, 0.75, 0.3, gate);
process = os.sawtooth(freq) : fi.resonbp(2, cut, q) * envelope * 0.7 <: _, _;
