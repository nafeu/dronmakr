declare name "Magma Flow";
declare description "Hot magma drone with slow filter motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.45, 0.75, 0.66, 1.8, gate);
cut = 900 * (1 + 0.35 * os.lf_triangle(0.08));
process = os.sawtooth(freq) : fi.lowpass(2, cut) * envelope * 0.6 <: _, _;
