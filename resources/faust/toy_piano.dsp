declare name "Toy Piano";
declare description "Simple bright toy piano tone.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.1, 0.45, 0.7, gate);
process = os.osc(freq) : fi.highpass(1, 120) * envelope <: _, _;
