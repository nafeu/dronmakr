declare name "Afterglow Pad";
declare description "Warm post-sunset glow with long tail.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.5, 0.85, 0.7, 1.9, gate);
process = (os.sawtooth(freq) : fi.lowpass(2, 1600)) * envelope * 0.65 <: _, _;
