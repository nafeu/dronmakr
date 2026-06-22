declare name "Deep Current";
declare description "Underwater current with sub motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.45, 0.85, 0.64, 2.1, gate);
voice = os.osc(freq*0.5) + no.noise : fi.lowpass(1, 260) * 0.2;
process = voice * envelope <: _, _;
