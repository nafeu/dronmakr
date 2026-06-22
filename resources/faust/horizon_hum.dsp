declare name "Horizon Hum";
declare description "Distant dual-sub horizon hum.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.5, 0.85, 0.68, 1.9, gate);
voice = os.osc(freq*0.5) + 0.7*os.osc(freq*0.25);
process = voice * envelope * 0.55 <: _, _;
