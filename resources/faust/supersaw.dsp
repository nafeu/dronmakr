declare name "Supersaw";
declare description "Triple detuned saws for a massive unison lead.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 14, 2, 40, 0.1);
envelope = gain * en.adsr(0.04, 0.18, 0.82, 0.35, gate);
voice = os.sawtooth(freq) + os.sawtooth(freq*(1+spread/900)) + os.sawtooth(freq*(1-spread/900));
process = voice * 0.33 * envelope <: _, _;
