declare name "Talkbox Saw";
declare description "Mid-focused resonant band on a bright saw.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 900, 200, 3500, 1);
q = hslider("q", 4.2, 1, 10, 0.01);
envelope = gain * en.adsr(0.02, 0.12, 0.7, 0.25, gate);
process = os.sawtooth(freq) : fi.resonbp(2, cut, q) * envelope * 0.65 <: _, _;
