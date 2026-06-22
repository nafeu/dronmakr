declare name "Ether Void";
declare description "Minimal long sine void with vast space.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.7, 1.1, 0.62, 2.4, gate);
process = os.osc(freq) * envelope * 0.8 <: _, _;
