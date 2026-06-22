declare name "Tectonic";
declare description "Slow sub tectonic rumble with weight.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.6, 1.0, 0.65, 2.2, gate);
voice = os.osc(freq*0.5) * 0.8 + os.sawtooth(freq) * 0.25;
process = voice : fi.lowpass(1, 420) * envelope <: _, _;
