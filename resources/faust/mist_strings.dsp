declare name "Mist Strings";
declare description "Distant filtered string ensemble.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.35, 0.6, 0.75, 1.3, gate);
voice = os.sawtooth(freq*1.004) + os.sawtooth(freq*0.996);
process = voice : fi.lowpass(2, 3000) * envelope * 0.5 <: _, _;
