declare name "Night Train";
declare description "Rhythmic pulsing drone like distant rails.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.35, 0.6, 0.62, 1.4, gate);
pulse = 0.55 + 0.45 * os.lf_triangle(0.35);
process = os.sawtooth(freq) : fi.lowpass(1, 700) * envelope * pulse * 0.65 <: _, _;
