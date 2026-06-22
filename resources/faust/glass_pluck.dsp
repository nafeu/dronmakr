declare name "Glass Pluck";
declare description "Brittle glassy pluck with fast decay.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.001, 0.32, gate);
process = os.triangle(freq) : fi.highpass(1, 300) * envelope * 0.7 <: _, _;
