declare name "Winter Hall";
declare description "Cold spacious hall pad with air.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.45, 0.75, 0.68, 1.7, gate);
process = os.triangle(freq) : fi.highpass(1, 180) : fi.lowpass(2, 5200) * envelope * 0.7 <: _, _;
