declare name "Nebula Pad";
declare description "Slow-blooming detuned space pad.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 9, 1, 24, 0.1);
envelope = gain * en.adsr(0.55, 0.8, 0.72, 1.8, gate);
voice = os.triangle(freq*(1+spread/1600)) + os.triangle(freq*(1-spread/1600));
process = voice * 0.42 * envelope <: _, _;
