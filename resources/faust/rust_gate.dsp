declare name "Rust Gate";
declare description "Gritty filtered square with a rusty edge.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 760, 150, 4200, 1);
envelope = gain * en.adsr(0.03, 0.14, 0.68, 0.28, gate);
process = os.square(freq) : fi.lowpass(2, cut) * envelope * 0.72 <: _, _;
