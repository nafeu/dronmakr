declare name "Silk Winds";
declare description "Airy detuned sine choir in motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 7, 1, 18, 0.1);
envelope = gain * en.adsr(0.35, 0.55, 0.78, 1.2, gate);
voice = os.osc(freq*(1+spread/2200)) + os.osc(freq) + os.osc(freq*(1-spread/2200));
process = voice * 0.3 * envelope <: _, _;
